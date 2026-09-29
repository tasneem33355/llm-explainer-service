"""
CrediX AI Explainability Layer
===============================
A thin, provider-agnostic layer that turns the structured JSON output of
CrediX's ML components -- the credit-risk PD scorer (app/model.py), the
5-layer fraud engine (fraud_engine.py), and the portfolio / risk-lab
analytics (portfolio_analytics.py, dashboard.py) -- into plain-language,
business-friendly explanations, and answers free-form follow-up questions
grounded strictly in that data.

Used by two front doors:
  - dashboard.py            -> "AI Assistant" chat tab (interactive, Streamlit)
  - api.py (/explain)       -> a stateless HTTP endpoint any system can call

Provider: Google Gemini, because it currently has the most usable free
tier (no credit card, generous daily quota) of the major hosted LLM APIs --
see docs/llm_layer.md for how to get a key. The only function that talks
to the network is `call_llm()`; swapping providers later (OpenAI, Claude,
a local model, ...) means rewriting that one function and leaving prompt
building, context formatting, and chat history untouched.

This module deliberately has ONE non-stdlib dependency (`requests`,
already used elsewhere in this repo) so it can be imported from both the
FastAPI services and the Streamlit dashboard without pulling in a heavy
SDK.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional

import requests

try:  # pragma: no cover - optional convenience, mirrors app/config.py
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

logger = logging.getLogger("credix.llm")
if not logger.handlers:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))


# ---------------------------------------------------------------------------
# Configuration (all overridable via environment / .env -- see .env.example)
# ---------------------------------------------------------------------------

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# gemini-2.5-flash / gemini-1.5-flash is Google's free-tier workhorse model
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

GEMINI_API_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent"
)

LLM_MAX_OUTPUT_TOKENS = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "1024"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
LLM_TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT_SECONDS", "30"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))
LLM_RETRY_BASE_DELAY_SECONDS = float(
    os.getenv("LLM_RETRY_BASE_DELAY_SECONDS", "2.0")
)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class LLMNotConfiguredError(RuntimeError):
    """Raised when the LLM layer is invoked without an API key configured."""


class LLMRequestError(RuntimeError):
    """Raised when an external call to the LLM provider fails permanently."""


# ---------------------------------------------------------------------------
# Language detection & prompt building
# ---------------------------------------------------------------------------

_ARABIC_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]")


def detect_language(text: Optional[str]) -> str:
    """Return 'ar' if text contains Arabic characters, otherwise 'en'."""
    if not text:
        return "en"
    return "ar" if _ARABIC_RE.search(text) else "en"


SYSTEM_PROMPT_EN = """You are CrediX Copilot, the AI risk & compliance assistant embedded in CrediX (an enterprise credit-risk underwriting platform for Egyptian banks and financial institutions).

Your role:
1. Explain credit scoring decisions (PD, credit score, risk tier, reason codes), fraud flags (anomalies, ghost employers, OOD), and portfolio analytics in clear, professional banking language.
2. Answer the underwriter's or risk manager's specific questions accurately.
3. Ground every statement STRICTLY in the provided JSON data. Never hallucinate facts, applicant data, financial figures, or regulatory rules that are not in the context.
4. When citing risk drivers, refer to the exact feature names, reason codes, or anomaly descriptions present in the input.
5. If the user asks something that cannot be answered from the provided data, state that clearly instead of guessing.
6. Keep answers concise, structured (use bullet points where appropriate), and tone professional and compliance-ready.
7. Always respond in English unless specifically instructed otherwise."""

SYSTEM_PROMPT_AR = """أنت CrediX Copilot، المساعد الذكي المدمج في منصة CrediX لإدارة مخاطر الائتمان ومكافحة الاحتيال في القطاع المصرفي المصري.

دورك:
1. شرح قرارات التقييم الائتماني (احتمالية التعثر PD، التقييم الائتماني، شريحة المخاطر، أكواد الأسباب)، ومؤشرات الاحتيال (الأنشطة غير الطبيعية، جهات العمل الوهمية، البيانات خارج التوزيع OOD)، وتحليلات المحفظة بلغة مصرفية واضحة واحترافية.
2. الإجابة بدقة على أسئلة مسؤول الائتمان أو مدير المخاطر.
3. الالتزام الصارم بالبيانات المتاحة في ملف الـ JSON المرفق. لا تختلق أرقاماً أو وقائع أو قواعد غير موجودة في السياق.
4. الإشارة إلى العوامل والمؤشرات بأسمائها وقيمها الفعلية الواردة في المدخلات.
5. إذا سأل المستخدم عن معلومة لا يمكن استنتاجها من البيانات المتاحة، وضح ذلك صراحةً بدلاً من التخمين.
6. اجعل الإجابات موجزة ومنظمة (استخدم النقاط عند الحاجة) وبأسلوب مهني يتماشى مع تقارير الالتزام والرقابة.
7. أجب دائماً باللغة العربية بأسلوب مصرفي سليم."""


def build_system_prompt(lang: str) -> str:
    return SYSTEM_PROMPT_AR if lang == "ar" else SYSTEM_PROMPT_EN


# ---------------------------------------------------------------------------
# Context serialization & prompt assembling
# ---------------------------------------------------------------------------

def _truncate_json(data: Any, max_chars: int = 8000) -> str:
    """Serialize data to JSON, truncating if it exceeds max_chars."""
    dumped = json.dumps(data, indent=2, ensure_ascii=False, default=str)
    if len(dumped) <= max_chars:
        return dumped
    return dumped[:max_chars] + "\n... [TRUNCATED FOR LENGTH]"


def build_context(block_type: str, data: Dict[str, Any]) -> str:
    """Format a single data block with a clear section header."""
    header_map = {
        "credit_risk": "=== CREDIT RISK SCORING RESULT ===",
        "fraud": "=== FRAUD & FORENSIC ASSESSMENT ===",
        "portfolio": "=== PORTFOLIO / RISK LAB ANALYTICS ===",
        "custom": "=== CONTEXT DATA ===",
    }
    header = header_map.get(block_type, f"=== {block_type.upper()} DATA ===")
    return f"{header}\n{_truncate_json(data)}"


def build_combined_context(blocks: List[Dict[str, Any]]) -> str:
    """Format multiple heterogeneous data blocks into a single prompt section."""
    sections = [
        build_context(b.get("type", "custom"), b.get("data", {}))
        for b in blocks
        if b.get("data")
    ]
    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Gemini API client
# ---------------------------------------------------------------------------

def _messages_to_gemini_contents(
    system_prompt: str,
    user_prompt: str,
    history: Optional[List[Dict[str, str]]] = None,
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Format system instruction and conversation turns for Gemini's generateContent endpoint.
    """
    system_instruction = {
        "parts": [{"text": system_prompt}]
    }

    contents: List[Dict[str, Any]] = []

    if history:
        for turn in history:
            role = turn.get("role")
            content = turn.get("content", "")
            if not content:
                continue
            gemini_role = "model" if role in ("assistant", "model") else "user"
            contents.append({
                "role": gemini_role,
                "parts": [{"text": content}]
            })

    contents.append({
        "role": "user",
        "parts": [{"text": user_prompt}]
    })

    return system_instruction, contents


def call_llm(
    system_prompt: str,
    user_prompt: str,
    history: Optional[List[Dict[str, str]]] = None,
) -> str:
    """
    Send prompt + history to Google Gemini generateContent with retries and exponential backoff.
    """
    if not GEMINI_API_KEY:
        raise LLMNotConfiguredError(
            "GEMINI_API_KEY is not set. Get a free key at "
            "https://aistudio.google.com/ and set it in your environment or .env file."
        )

    system_instruction, contents = _messages_to_gemini_contents(
        system_prompt, user_prompt, history
    )

    payload = {
        "system_instruction": system_instruction,
        "contents": contents,
        "generationConfig": {
            "temperature": LLM_TEMPERATURE,
            "maxOutputTokens": LLM_MAX_OUTPUT_TOKENS,
        },
    }

    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": GEMINI_API_KEY,
    }

    delay = LLM_RETRY_BASE_DELAY_SECONDS
    last_err_msg: str = "unknown"

    for attempt in range(1, LLM_MAX_RETRIES + 1):
        try:
            logger.info("Calling Gemini (attempt %d/%d) model=%s", attempt, LLM_MAX_RETRIES, GEMINI_MODEL)
            response = requests.post(
                GEMINI_API_URL,
                json=payload,
                headers=headers,
                timeout=LLM_TIMEOUT_SECONDS,
            )

            if response.status_code == 200:
                body = response.json()
                try:
                    candidates = body.get("candidates", [])
                    if not candidates:
                        raise LLMRequestError(f"Gemini returned no candidates: {body}")
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if not parts or "text" not in parts[0]:
                        raise LLMRequestError(f"Gemini response missing text part: {body}")
                    return parts[0]["text"]
                except (KeyError, IndexError) as parse_err:
                    raise LLMRequestError(
                        f"Failed to parse Gemini response: {body}"
                    ) from parse_err

            if response.status_code in (429, 500, 503):
                last_err_msg = f"HTTP {response.status_code}: {response.text[:300]}"
                logger.warning(
                    "Gemini retryable status %d on attempt %d/%d: %s. Retrying in %.1fs...",
                    response.status_code,
                    attempt,
                    LLM_MAX_RETRIES,
                    response.text[:300],
                    delay,
                )
                time.sleep(delay)
                delay *= 2.0
                continue

            # Non-retryable HTTP error (400, 401, 403, 404)
            error_detail = f"Gemini API error {response.status_code}: {response.text[:500]}"
            logger.error("Non-retryable Gemini error: %s", error_detail)
            raise LLMRequestError(error_detail)

        except requests.RequestException as req_err:
            last_err_msg = str(req_err)
            logger.warning(
                "Gemini network failure on attempt %d/%d: %s. Retrying in %.1fs...",
                attempt,
                LLM_MAX_RETRIES,
                req_err,
                delay,
            )
            time.sleep(delay)
            delay *= 2.0

    raise LLMRequestError(
        f"Gemini call failed after {LLM_MAX_RETRIES} attempts. Last error: {last_err_msg}"
    )


# ---------------------------------------------------------------------------
# High-level entrypoint
# ---------------------------------------------------------------------------

def explain_result(
    blocks: List[Dict[str, Any]],
    question: Optional[str] = None,
    lang: Optional[str] = None,
    history: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, str]:
    """
    Main entry point for generating explanations or answering questions.
    """
    if not lang:
        lang = detect_language(question) if question else "en"

    system_prompt = build_system_prompt(lang)
    context_str = build_combined_context(blocks)

    if not context_str.strip():
        context_str = "No structured result data provided."

    if question and question.strip():
        if lang == "ar":
            user_prompt = (
                f"فيما يلي نتائج التحليل:\n\n{context_str}\n\n"
                f"سؤال المستخدم:\n{question}\n\n"
                f"أجب عن السؤال استناداً فقط إلى البيانات أعلاه وبأسلوب مصرفي مهني."
            )
        else:
            user_prompt = (
                f"Below is the analysis result data:\n\n{context_str}\n\n"
                f"User question:\n{question}\n\n"
                f"Answer the question grounded strictly in the data above, in professional banking terms."
            )
    else:
        if lang == "ar":
            user_prompt = (
                f"فيما يلي نتائج التحليل للطلب:\n\n{context_str}\n\n"
                f"المطلوب: قدم ملخصاً تنفيذياً شاملاً يشرح:\n"
                f"1. القرار النهائي ومستوى المخاطر.\n"
                f"2. العوامل الرئيسية الدافعة للقرار (نقاط القوة والمخاطر المحددة).\n"
                f"3. أي توصيات لإجراءات تالية أو فحوصات إضافية.\n"
                f"اجعل الأسلوب رسمياً وموجزاً ومناسباً لتقرير ائتماني."
            )
        else:
            user_prompt = (
                f"Below is the analysis result data for the application:\n\n{context_str}\n\n"
                f"Provide a comprehensive executive summary covering:\n"
                f"1. Final decision and risk classification.\n"
                f"2. Key drivers (primary strengths and identified risks/anomalies).\n"
                f"3. Recommended next steps or manual verification items.\n"
                f"Keep the tone formal, concise, and audit-ready."
            )

    answer = call_llm(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        history=history,
    )

    return {
        "answer": answer,
        "language": lang,
    }
