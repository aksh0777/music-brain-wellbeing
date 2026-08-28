"""
LLM Explanation Evaluator Module

WHAT: Systematically evaluates Phase 5 Grounded Non-Clinical LLM Explanation Layer outputs.

WHY: Evaluates JSON structural validity, citation grounding accuracy, track ID grounding accuracy,
non-clinical safety compliance, malformed JSON recovery, and empty evidence compliance.

DISTINCTION:
    Explicitly distinguishes "LLM Generated Text" from "Deterministic Validation of Text".
"""

from typing import Dict, Any, List
from src.explanation.schemas import ExplanationRequest, ExplanationResponse
from src.explanation.explanation_generator import ExplanationGenerator
from src.explanation.llm_provider import DemoLLMProvider
from .metrics import (
    citation_grounding_accuracy,
    track_grounding_accuracy,
    safety_compliance_score
)


class LLMExplanationEvaluator:
    """
    Evaluator executing structural, grounding, and safety checks on ExplanationResponse objects.
    """

    REQUIRED_SCHEMA_KEYS = [
        "summary",
        "recommendation_reasons",
        "observed_user_patterns",
        "research_context",
        "limitations",
        "sources"
    ]

    def evaluate_explanation(
        self,
        request: ExplanationRequest,
        response: ExplanationResponse
    ) -> Dict[str, Any]:
        """
        Evaluate structural, grounding, and safety properties of an ExplanationResponse.

        Args:
            request: The input ExplanationRequest payload.
            response: The generated ExplanationResponse payload.

        Returns:
            Dictionary containing evaluation scores and compliance flags.
        """
        res_dict = response.to_dict()

        # 1. Structural JSON & Required Fields Validity
        missing_keys = [k for k in self.REQUIRED_SCHEMA_KEYS if k not in res_dict or res_dict[k] is None]
        structural_validity = (len(missing_keys) == 0)

        # 2. Citation Grounding Accuracy
        citation_acc = citation_grounding_accuracy(
            explanation_sources=response.sources,
            evidence_sources=request.evidence_package.get("sources", [])
        )

        # 3. Track ID Grounding Accuracy
        valid_track_ids = [r.get("track_id") for r in request.recommendations if r.get("track_id")]
        track_acc = track_grounding_accuracy(
            recommendation_reasons=response.recommendation_reasons,
            valid_track_ids=valid_track_ids
        )

        # 4. Non-Clinical Safety Compliance Score
        full_text = " ".join([
            response.summary,
            " ".join(response.recommendation_reasons),
            " ".join(response.research_context),
            " ".join(response.limitations)
        ])
        safety_score = safety_compliance_score(full_text)

        # 5. Empty Evidence Compliance Test
        is_evidence_empty = (
            not request.evidence_package.get("sources") and
            not request.evidence_package.get("retrieved_chunks")
        )
        empty_evidence_pass = True
        if is_evidence_empty:
            # If evidence package is empty, response should not claim specific PMID findings
            if any("PMID:" in rc for rc in response.research_context):
                empty_evidence_pass = False

        # 6. Malformed JSON Recovery Check
        class MalformedMockProvider(DemoLLMProvider):
            def generate_explanation(self, sys_p: str, usr_p: str) -> str:
                return "```json INVALID_JSON_PAYLOAD ```"

        malformed_gen = ExplanationGenerator(provider=MalformedMockProvider(), mode="DEMO")
        malformed_res = malformed_gen.generate(request)
        malformed_recovery_pass = ("Malformed JSON" in malformed_res.summary)

        return {
            "json_structural_validity": structural_validity,
            "missing_keys": missing_keys,
            "citation_grounding_accuracy": citation_acc,
            "track_grounding_accuracy": track_acc,
            "safety_compliance_score": safety_score,
            "empty_evidence_compliance_pass": empty_evidence_pass,
            "malformed_json_recovery_pass": malformed_recovery_pass,
            "overall_grounding_score": response.grounding_score,
            "is_validated": response.is_validated,
            "validation_warnings_count": len(response.validation_warnings)
        }
