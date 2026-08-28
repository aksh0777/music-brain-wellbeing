"""
Unit Tests for Phase 5 Grounded Non-Clinical LLM Explanation Layer

WHAT: Comprehensive test suite validating explanation schema data contracts, prompt construction,
safety constraints, provider abstraction, DemoLLMProvider offline execution, PMID/DOI citation validation,
track ID grounding, missing evidence handling, malformed JSON recovery, and end-to-end pipeline coordination.

WHY: Ensures 100% deterministic, offline verification without external LLM API dependencies.
"""

import unittest
import json
from src.explanation.schemas import SafetyConstraints, ExplanationRequest, ExplanationResponse
from src.explanation.llm_provider import DemoLLMProvider, GenericOpenAILLMProvider
from src.explanation.prompt import build_system_prompt, build_user_prompt
from src.explanation.validation import GroundingValidator
from src.explanation.explanation_generator import ExplanationGenerator


class TestExplanationLayer(unittest.TestCase):

    def setUp(self):
        """Set up fixture data contracts for testing."""
        self.sample_user_profile = {
            "user_id": "usr_test_999",
            "total_tracks_listened": 42,
            "avg_tracks_per_session": 6.5,
            "data_provenance": "demo"
        }

        self.sample_recommendations = [
            {
                "track_id": "TRK_101",
                "track_name": "Weightless Ambient",
                "similarity_score": 0.92,
                "final_score": 0.88
            },
            {
                "track_id": "TRK_102",
                "track_name": "Calm Waters",
                "similarity_score": 0.86,
                "final_score": 0.82
            }
        ]

        self.sample_acoustic_profiles = {
            "user_acoustic_mean": {"energy": 0.35, "tempo": 85.0},
            "track_acoustic_match": "Low Energy Soothing Acoustics"
        }

        self.sample_evidence_package = {
            "query": "music therapy stress reduction RCT meta-analysis",
            "retrieved_chunks": [
                {
                    "chunk_id": "pub_dewitte_2022_33176590_chunk_0",
                    "text": "A meta-analysis evaluating music therapy for stress reduction.",
                    "metadata": {
                        "document_id": "pub_dewitte_2022_33176590",
                        "title": "Music therapy for stress reduction",
                        "authors_str": "de Witte et al.",
                        "year": 2022,
                        "pmid": "33176590",
                        "doi": "10.1080/17437199.2020.1846580"
                    }
                }
            ],
            "sources": [
                {
                    "document_id": "pub_dewitte_2022_33176590",
                    "title": "Music therapy for stress reduction",
                    "authors": "de Witte et al.",
                    "year": 2022,
                    "pmid": "33176590",
                    "doi": "10.1080/17437199.2020.1846580"
                }
            ],
            "retrieval_metadata": {
                "total_chunks_retrieved": 1,
                "total_distinct_sources": 1
            }
        }

        self.sample_request = ExplanationRequest(
            user_profile=self.sample_user_profile,
            recommendations=self.sample_recommendations,
            acoustic_profiles=self.sample_acoustic_profiles,
            evidence_package=self.sample_evidence_package
        )

    def test_1_explanation_request_construction(self):
        """Test ExplanationRequest schema serialization and structure."""
        req_dict = self.sample_request.to_dict()
        self.assertIn("user_profile", req_dict)
        self.assertIn("recommendations", req_dict)
        self.assertIn("evidence_package", req_dict)
        self.assertIn("safety_constraints", req_dict)
        self.assertEqual(len(req_dict["recommendations"]), 2)

    def test_2_explanation_response_serialization(self):
        """Test ExplanationResponse to_dict and from_dict roundtrip."""
        res = ExplanationResponse(
            summary="Test summary",
            recommendation_reasons=["Reason 1"],
            observed_user_patterns=["Pattern 1"],
            research_context=["Context 1"],
            limitations=["Limitation 1"],
            sources=[{"title": "Paper A", "pmid": "12345"}]
        )
        d = res.to_dict()
        self.assertEqual(d["summary"], "Test summary")
        res_reconstructed = ExplanationResponse.from_dict(d)
        self.assertEqual(res_reconstructed.summary, "Test summary")
        self.assertEqual(res_reconstructed.sources[0]["pmid"], "12345")

    def test_3_system_prompt_safety_rules(self):
        """Test that build_system_prompt includes required non-clinical safety guardrails."""
        sys_prompt = build_system_prompt()
        self.assertIn("NON-CLINICAL SAFETY BOUNDARY", sys_prompt)
        self.assertIn("MUST NOT diagnose anxiety", sys_prompt)
        self.assertIn("NEVER invent citations", sys_prompt)
        self.assertIn("CORRELATION/ASSOCIATION from CAUSATION", sys_prompt)

    def test_4_user_prompt_formatting(self):
        """Test that build_user_prompt generates valid JSON payload containing request context."""
        user_prompt = build_user_prompt(self.sample_request)
        parsed = json.loads(user_prompt)
        self.assertEqual(parsed["user_profile"]["user_id"], "usr_test_999")
        self.assertEqual(len(parsed["recommendations"]), 2)

    def test_5_demo_llm_provider_offline_execution(self):
        """Test DemoLLMProvider generates valid JSON output offline without API keys."""
        provider = DemoLLMProvider()
        sys_prompt = build_system_prompt()
        user_prompt = build_user_prompt(self.sample_request)
        raw_out = provider.generate_explanation(sys_prompt, user_prompt)

        parsed = json.loads(raw_out)
        self.assertIn("summary", parsed)
        self.assertIn("recommendation_reasons", parsed)
        self.assertIn("research_context", parsed)

    def test_6_valid_citation_validation(self):
        """Test GroundingValidator marks response valid when cited PMID/DOI exists in EvidencePackage."""
        validator = GroundingValidator()
        res = ExplanationResponse(
            summary="Valid explanation",
            recommendation_reasons=["Matches track ID: TRK_101"],
            observed_user_patterns=["Average 6.5 tracks per session."],
            research_context=["Research by de Witte (PMID: 33176590) investigated stress reduction."],
            limitations=["Non-clinical disclaimer."],
            sources=[{"title": "Music therapy for stress reduction", "pmid": "33176590", "doi": "10.1080/17437199.2020.1846580"}]
        )
        validated = validator.validate(self.sample_request, res)
        self.assertTrue(validated.is_validated)
        self.assertEqual(validated.grounding_score, 1.0)
        self.assertEqual(len(validated.validation_warnings), 0)

    def test_7_fabricated_citation_detection(self):
        """Test GroundingValidator flags warning and reduces grounding score when PMID/DOI is fabricated."""
        validator = GroundingValidator()
        res = ExplanationResponse(
            summary="Explanation with fake source",
            recommendation_reasons=["Matches track ID: TRK_101"],
            observed_user_patterns=["Pattern A"],
            research_context=["Fake paper claims PMID: 9999999."],
            limitations=["Limitation A"],
            sources=[{"title": "Fake Paper", "pmid": "9999999"}]
        )
        validated = validator.validate(self.sample_request, res)
        self.assertFalse(validated.is_validated)
        self.assertLess(validated.grounding_score, 1.0)
        self.assertTrue(any("Fabricated PMID" in w or "ungrounded PMID" in w for w in validated.validation_warnings))

    def test_8_track_id_validation(self):
        """Test GroundingValidator flags warning when recommendation reason references ungrounded track ID."""
        validator = GroundingValidator()
        res = ExplanationResponse(
            summary="Explanation with invalid track",
            recommendation_reasons=["Selected track ID: TRK_FABRICATED_999"],
            observed_user_patterns=["Pattern A"],
            research_context=["Context A"],
            limitations=["Limitation A"],
            sources=[]
        )
        validated = validator.validate(self.sample_request, res)
        self.assertTrue(any("ungrounded track ID" in w for w in validated.validation_warnings))

    def test_9_empty_evidence_handling(self):
        """Test validator flags warning if EvidencePackage is empty but response claims specific findings."""
        empty_evidence_request = ExplanationRequest(
            user_profile=self.sample_user_profile,
            recommendations=self.sample_recommendations,
            acoustic_profiles=self.sample_acoustic_profiles,
            evidence_package={"query": "test", "retrieved_chunks": [], "sources": [], "retrieval_metadata": {"total_chunks_retrieved": 0}}
        )
        validator = GroundingValidator()
        res = ExplanationResponse(
            summary="Empty evidence test",
            recommendation_reasons=["Reason A"],
            observed_user_patterns=["Pattern A"],
            research_context=["Research investigated stress reduction with PMID: 33176590."],
            limitations=["Limitation A"],
            sources=[]
        )
        validated = validator.validate(empty_evidence_request, res)
        self.assertTrue(any("EvidencePackage is empty" in w for w in validated.validation_warnings))

    def test_10_safety_constraint_enforcement(self):
        """Test validator flags warning if response contains prohibited clinical terms."""
        validator = GroundingValidator()
        res = ExplanationResponse(
            summary="This music will diagnose anxiety and provide a clinical treatment cure.",
            recommendation_reasons=["Reason A"],
            observed_user_patterns=["Pattern A"],
            research_context=["Context A"],
            limitations=["Limitation A"],
            sources=[]
        )
        validated = validator.validate(self.sample_request, res)
        self.assertFalse(validated.is_validated)
        self.assertTrue(any("Prohibited non-clinical safety term" in w for w in validated.validation_warnings))

    def test_11_malformed_json_handling(self):
        """Test ExplanationGenerator handles malformed JSON string gracefully without crashing."""
        class MalformedProvider(DemoLLMProvider):
            def generate_explanation(self, system_prompt: str, user_prompt: str) -> str:
                return "THIS IS NOT VALID JSON {malformed..."

        generator = ExplanationGenerator(provider=MalformedProvider(), mode="DEMO")
        res = generator.generate(self.sample_request)
        self.assertIsInstance(res, ExplanationResponse)
        self.assertIn("Malformed JSON", res.summary)

    def test_12_end_to_end_generator_execution(self):
        """Test end-to-end ExplanationGenerator execution flow in DEMO mode."""
        generator = ExplanationGenerator(mode="DEMO")
        res = generator.generate(self.sample_request)

        self.assertIsInstance(res, ExplanationResponse)
        self.assertTrue(res.is_validated)
        self.assertEqual(res.grounding_score, 1.0)
        self.assertGreater(len(res.recommendation_reasons), 0)
        self.assertGreater(len(res.sources), 0)


if __name__ == "__main__":
    unittest.main()
