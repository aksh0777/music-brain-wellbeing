"""
Explanation Layer Schemas & Data Contracts Module

WHAT: Defines strongly typed schema data contracts (`SafetyConstraints`, `ExplanationRequest`, `ExplanationResponse`)
for the LLM Explanation Layer.

WHY: Enforces strict operational data contracts between upstream recommendation/RAG outputs and downstream LLM generation.
Prevents passing unstructured application state to LLMs and ensures deterministic, JSON-serializable output for backend serving.

DATA CONTRACT HIERARCHY:
    User Profile + Recommendations + Acoustic Profile + RAG Evidence Package + Safety Constraints
                                           ↓
                                  ExplanationRequest
                                           ↓
                                  LLM Provider & Prompt
                                           ↓
                                  ExplanationResponse
"""

from typing import Dict, Any, List, Optional
import json


class SafetyConstraints:
    """
    Immutable non-clinical safety rules and scientific boundaries enforced during prompt building and output validation.
    """

    def __init__(
        self,
        prohibit_clinical_diagnosis: bool = True,
        prohibit_treatment_claims: bool = True,
        prohibit_fabricated_citations: bool = True,
        require_non_clinical_disclaimer: bool = True,
        require_causation_disclaimer: bool = True
    ):
        self.prohibit_clinical_diagnosis = prohibit_clinical_diagnosis
        self.prohibit_treatment_claims = prohibit_treatment_claims
        self.prohibit_fabricated_citations = prohibit_fabricated_citations
        self.require_non_clinical_disclaimer = require_non_clinical_disclaimer
        self.require_causation_disclaimer = require_causation_disclaimer

    def to_dict(self) -> Dict[str, bool]:
        return {
            "prohibit_clinical_diagnosis": self.prohibit_clinical_diagnosis,
            "prohibit_treatment_claims": self.prohibit_treatment_claims,
            "prohibit_fabricated_citations": self.prohibit_fabricated_citations,
            "require_non_clinical_disclaimer": self.require_non_clinical_disclaimer,
            "require_causation_disclaimer": self.require_causation_disclaimer
        }


class ExplanationRequest:
    """
    Structured input request payload passed to the LLM explanation generator.
    """

    def __init__(
        self,
        user_profile: Dict[str, Any],
        recommendations: List[Dict[str, Any]],
        acoustic_profiles: Dict[str, Any],
        evidence_package: Dict[str, Any],
        safety_constraints: Optional[SafetyConstraints] = None
    ):
        self.user_profile = user_profile
        self.recommendations = recommendations
        self.acoustic_profiles = acoustic_profiles
        self.evidence_package = evidence_package
        self.safety_constraints = safety_constraints if safety_constraints is not None else SafetyConstraints()

    def to_dict(self) -> Dict[str, Any]:
        """Convert request payload into a serializable JSON-compatible dictionary."""
        return {
            "user_profile": self.user_profile,
            "recommendations": self.recommendations,
            "acoustic_profiles": self.acoustic_profiles,
            "evidence_package": self.evidence_package,
            "safety_constraints": self.safety_constraints.to_dict()
        }


class ExplanationResponse:
    """
    Structured response payload returned by the LLM explanation generator after grounding validation.
    """

    def __init__(
        self,
        summary: str,
        recommendation_reasons: List[str],
        observed_user_patterns: List[str],
        research_context: List[str],
        limitations: List[str],
        sources: List[Dict[str, Any]],
        is_validated: bool = False,
        validation_warnings: Optional[List[str]] = None,
        grounding_score: float = 1.0
    ):
        self.summary = summary
        self.recommendation_reasons = recommendation_reasons
        self.observed_user_patterns = observed_user_patterns
        self.research_context = research_context
        self.limitations = limitations
        self.sources = sources
        self.is_validated = is_validated
        self.validation_warnings = validation_warnings if validation_warnings is not None else []
        self.grounding_score = grounding_score

    def to_dict(self) -> Dict[str, Any]:
        """Convert response payload into a serializable JSON-compatible dictionary."""
        return {
            "summary": self.summary,
            "recommendation_reasons": self.recommendation_reasons,
            "observed_user_patterns": self.observed_user_patterns,
            "research_context": self.research_context,
            "limitations": self.limitations,
            "sources": self.sources,
            "is_validated": self.is_validated,
            "validation_warnings": self.validation_warnings,
            "grounding_score": self.grounding_score
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExplanationResponse":
        """Construct ExplanationResponse instance from a dictionary payload."""
        return cls(
            summary=data.get("summary", ""),
            recommendation_reasons=data.get("recommendation_reasons", []),
            observed_user_patterns=data.get("observed_user_patterns", []),
            research_context=data.get("research_context", []),
            limitations=data.get("limitations", []),
            sources=data.get("sources", []),
            is_validated=data.get("is_validated", False),
            validation_warnings=data.get("validation_warnings", []),
            grounding_score=data.get("grounding_score", 1.0)
        )
