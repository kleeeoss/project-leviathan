from typing import List, Dict, Type
from pydantic import BaseModel, Field


class TechSummary(BaseModel):
    service_name: str = Field(description="Official name of the service or technology")
    summary: str = Field(description="2-sentence architectural summary")
    key_characteristics: List[str] = Field(default_factory=list, description="Key architectural traits or strengths")


class ComparisonOutput(BaseModel):
    comparison_summary: str = Field(description="Comparative analysis of the technologies")
    key_differences: List[str] = Field(default_factory=list, description="Key operational or architectural differences")
    recommendation: str = Field(description="Concluding recommendation or takeaway")


SCHEMA_REGISTRY: Dict[str, Type[BaseModel]] = {
    "TechSummary": TechSummary,
    "ComparisonOutput": ComparisonOutput,
}
