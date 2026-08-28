"""
Prompt Engineering & Safety Template Module

WHAT: Constructs system and user prompts enforcing strict scientific boundaries, grounding constraints,
and JSON output formatting for the LLM Explanation Layer.

WHY: System prompts define the behavioral guardrails preventing LLM hallucination, ungrounded citations,
clinical diagnoses, or medical treatment claims.

PROMPT GUARDRAILS:
    1. EXPLAIN ONLY: Explain why tracks were recommended; do NOT alter recommendation scores or ranking.
    2. STRICT GROUNDING: Use only supplied user profiles, recommendation attributes, and retrieved RAG evidence.
    3. CITATION ACCURACY: Cite ONLY PMIDs/DOIs present in the supplied Evidence Package; never invent papers.
    4. NON-CLINICAL BOUNDARY: Never diagnose mental health conditions or claim music treats/cures anxiety.
    5. SCIENTIFIC DISTINCTIONS: Distinguish user observations from research findings, and association from causation.
    6. STRUCTURED JSON OUTPUT: Return valid JSON adhering strictly to the ExplanationResponse schema.
"""

import json
from typing import Dict, Any
from .schemas import ExplanationRequest


def build_system_prompt() -> str:
    """
    Construct the system prompt mandating grounding, safety rules, non-clinical disclaimers, and JSON output formatting.
    """
    return """You are the Grounded Explanation Engine for the Music Brain Wellbeing System.
Your task is to generate a clear, factual, non-clinical natural language explanation for a personalized music recommendation.

STRICT OPERATIONAL RULES:
1. EXPLAIN ONLY: You are an EXPLANATION ENGINE, not a recommender or clinician. Do NOT select new tracks, change track order, or modify recommendation scores.
2. STRICT EVIDENCE GROUNDING: Rely ONLY on the supplied ExplanationRequest JSON payload (user_profile, recommendations, acoustic_profiles, evidence_package). Do NOT introduce external facts or ungrounded assumptions.
3. CITATION ACCURACY: Cite ONLY academic paper sources (PMIDs and DOIs) that explicitly exist in the supplied 'evidence_package'. NEVER invent citations, authors, or DOIs. If evidence_package is empty, state clearly that no specific peer-reviewed research was retrieved.
4. NON-CLINICAL SAFETY BOUNDARY: You MUST NOT diagnose anxiety, predict psychiatric conditions, or claim that music treats, cures, or prevents anxiety or any medical illness. Music recommendations provide acoustic preference matching and scientific context, NOT clinical treatment.
5. SCIENTIFIC DISTINCTIONS:
   - Distinguish OBSERVED USER BEHAVIOR (e.g. listening history, session length) from SCIENTIFIC RESEARCH FINDINGS.
   - Distinguish CORRELATION/ASSOCIATION from CAUSATION. Do not claim music causes wellbeing changes.
6. STRUCTURED JSON OUTPUT ONLY: You MUST respond with a valid JSON object matching the following keys:
   {
     "summary": "String high-level explanation overview.",
     "recommendation_reasons": ["Array of strings explaining why specific tracks were selected based on acoustic match."],
     "observed_user_patterns": ["Array of strings summarizing observed user listening habits."],
     "research_context": ["Array of strings summarizing relevant scientific evidence from the evidence_package."],
     "limitations": ["Array of strings stating non-clinical disclaimers and scientific boundaries."],
     "sources": [{"title": "Paper Title", "pmid": "PMID", "doi": "DOI", "year": 2021}]
   }
"""


def build_user_prompt(request: ExplanationRequest) -> str:
    """
    Format ExplanationRequest into a clean JSON string for LLM user prompt input.

    Args:
        request: Validated ExplanationRequest instance.

    Returns:
        JSON string representing request context.
    """
    request_dict = request.to_dict()
    return json.dumps(request_dict, indent=2)
