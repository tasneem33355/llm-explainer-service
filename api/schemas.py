"""
CrediX AI Explainer - Pydantic Request & Response Schemas
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class ExplainBlock(BaseModel):
    type: str = Field(..., description="One of: credit_risk, fraud, portfolio, custom")
    data: Dict[str, Any]


class ExplainRequest(BaseModel):
    blocks: List[ExplainBlock] = Field(..., description="Result data to explain")
    question: Optional[str] = Field(default=None, description="Follow-up question or omit for auto summary")
    lang: Optional[str] = Field(default=None, description="'ar' or 'en'")
    history: Optional[List[Dict[str, str]]] = Field(default=None, description="Prior chat history")


class ExplainResponse(BaseModel):
    answer: str
    language: str
