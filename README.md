# CrediX AI Explainer Service

Enterprise explainability microservice translating complex credit scoring decisions, forensic fraud signals, and risk analytics into plain-language, audit-compliant English and Arabic narratives.

## Repository Architecture

```text
llm-explainer-service/
├── api/
│   ├── __init__.py
│   ├── main.py              # FastAPI entrypoint, CORS configuration, health checks
│   ├── schemas.py           # Pydantic contracts (ExplainBlock, ExplainRequest, ExplainResponse)
│   └── routes/
│       ├── __init__.py
│       └── explain.py       # POST /explain endpoint with error handling
├── src/
│   ├── __init__.py
│   └── explainer.py         # Prompt engineering & Gemini free-tier inference engine
├── tests/
│   ├── __init__.py
│   └── test_explainer.py    # Automated test suite
├── requirements.txt         # Production dependencies
├── Dockerfile               # Container build configuration
├── .gitignore               # Version control ignore rules
└── README.md                # Service documentation
```

## API Specifications

### 1. Health Status
- **Method:** `GET`
- **Endpoint:** `/health`
- **Description:** Verifies whether `GEMINI_API_KEY` is configured without making external network calls.

### 2. Generate Decision Narrative / Answer Questions
- **Method:** `POST`
- **Endpoint:** `/explain`
- **Request Body:**
```json
{
  "blocks": [
    {
      "type": "credit_risk",
      "data": {
        "decision": "REJECT",
        "credit_score": 540,
        "default_probability": 0.185,
        "risk_tier": "High Risk",
        "reason_codes": [
          "DTI_RATIO_EXCEEDS_50_PERCENT",
          "ISCORE_DELINQUENCY_OVER_60_DAYS"
        ]
      }
    },
    {
      "type": "fraud",
      "data": {
        "fraud_risk_score": 0.72,
        "fraud_risk_level": "HIGH",
        "recommended_action": "ESCALATE_TO_FRAUD_TEAM"
      }
    }
  ],
  "question": null,
  "lang": "ar",
  "history": []
}
```

- **Response (HTTP 200):**
```json
{
  "answer": "تم رفض طلب التمويل نتيجة تجاوز نسبة عبء الدين (DTI) الحد المسموح به رقابياً، بالإضافة إلى وجود متأخرات سابقة في السجل الائتماني (آي سكور) تجاوزت 60 يوماً. يوصى بإحالة الملف لفريق مكافحة الاحتيال لمراجعة المؤشرات المرتفعة.",
  "language": "ar"
}
```

## Environment Variables

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `GEMINI_API_KEY` | *(Required for inference)* | Free API key from https://aistudio.google.com/ |
| `LLM_PROVIDER` | `gemini` | LLM backend provider |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Gemini model name |
| `PORT` | `8000` | Microservice listening port |
| `LLM_MAX_OUTPUT_TOKENS` | `1024` | Maximum token length for generated narratives |
| `LLM_TEMPERATURE` | `0.2` | Deterministic sampling temperature |

## Quick Start (Local)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set API key
export GEMINI_API_KEY="your-gemini-key"

# 3. Start the service
uvicorn api.main.py:app --host 0.0.0.0 --port 8000 --reload

# 4. View documentation
# Open http://localhost:8000/docs
```

## Running Tests

```bash
pytest tests/
```

## Container Deployment

```bash
docker build -t llm-explainer-service .
docker run -p 8000:8000 -e GEMINI_API_KEY="your-gemini-key" llm-explainer-service
```
