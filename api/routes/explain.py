"""
Explain Route Handler
Exposes POST /explain to generate plain-language explanations of model decisions.
"""

from fastapi import APIRouter, HTTPException, status
from api.schemas import ExplainRequest, ExplainResponse
from src.explainer import (
    explain_result,
    LLMNotConfiguredError,
    LLMRequestError,
)

router = APIRouter(tags=["AI Explainability"])


@router.post("/explain", response_model=ExplainResponse, summary="Explain credit, fraud, or portfolio results")
def explain(payload: ExplainRequest):
    """
    Translates complex underwriting decisions, fraud signals, and risk analytics
    into clear, audit-compliant English and Arabic narratives.
    """
    try:
        result = explain_result(
            blocks=[b.model_dump() for b in payload.blocks],
            question=payload.question,
            lang=payload.lang,
            history=payload.history,
        )
        return ExplainResponse(
            answer=result["answer"],
            language=result["language"]
        )
    except LLMNotConfiguredError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))
    except LLMRequestError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected explainer error: {str(exc)}"
        )
