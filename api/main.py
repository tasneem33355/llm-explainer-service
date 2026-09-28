"""
CrediX AI Explainer Microservice - Main Application Entrypoint
"""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes.explain import router as explain_router

app = FastAPI(
    title="CrediX AI Explainer Service",
    version="1.0.0",
    description="LLM Explainer Microservice translating underwriting and fraud decisions into plain English and Arabic.",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS Middleware configured to allow Vercel frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://credi-x.vercel.app",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "*",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the explain route
app.include_router(explain_router)


@app.get("/", tags=["Health"])
def root():
    return {
        "service": "llm-explainer-service",
        "status": "online",
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"])
def health_check():
    """
    Returns whether the LLM API key is configured without making network calls.
    """
    api_key_configured = bool(os.getenv("GEMINI_API_KEY", "").strip())
    provider = os.getenv("LLM_PROVIDER", "gemini").lower()
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    return {
        "status": "healthy",
        "llm_provider": provider,
        "model": model_name,
        "api_key_configured": api_key_configured,
        "environment": os.getenv("ENVIRONMENT", "production"),
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("api.main.py:app", host="0.0.0.0", port=port, reload=False)
