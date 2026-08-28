"""
LLM Explanation Pipeline Coordinator Module

WHAT: Orchestrates the complete Phase 5 explanation workflow:
      ExplanationRequest → Prompt Construction → LLM Provider Execution → JSON Parsing → Grounding Validation → ExplanationResponse.

WHY: Provides a single, clean entry-point for the explanation layer. Decouples prompt formatting,
LLM provider selection, JSON parsing, and grounding validation into a robust coordinator.

PIPELINE FLOW:
    ExplanationRequest Payload
                ↓
    build_system_prompt() & build_user_prompt()
                ↓
    LLMProvider (DemoLLMProvider or GenericOpenAILLMProvider)
                ↓
    JSON Parsing & Error Recovery
                ↓
    GroundingValidator (Citation, Track ID & Safety Rules)
                ↓
    Validated ExplanationResponse Object
"""

import os
import json
from typing import Dict, Any, Optional
from .schemas import ExplanationRequest, ExplanationResponse
from .llm_provider import LLMProvider, DemoLLMProvider, GenericOpenAILLMProvider
from .prompt import build_system_prompt, build_user_prompt
from .validation import GroundingValidator


class ExplanationGenerator:
    """
    Coordinator managing prompt assembly, LLM execution, JSON parsing, and grounding validation.
    """

    def __init__(
        self,
        provider: Optional[LLMProvider] = None,
        validator: Optional[GroundingValidator] = None,
        mode: Optional[str] = None
    ):
        """
        Initialize ExplanationGenerator.

        Args:
            provider: Custom LLMProvider instance. If None, selects based on mode / env vars.
            validator: Custom GroundingValidator instance. If None, creates default.
            mode: 'DEMO' or 'REAL'. Defaults to env var LLM_MODE or 'DEMO'.
        """
        self.mode = (mode or os.getenv("LLM_MODE", "DEMO")).upper()
        self.validator = validator if validator is not None else GroundingValidator()

        if provider is not None:
            self.provider = provider
        else:
            api_key = os.getenv("LLM_API_KEY", "").strip()
            if self.mode == "REAL" and api_key:
                self.provider = GenericOpenAILLMProvider()
            else:
                # Default to DEMO mode for offline/test reliability
                self.provider = DemoLLMProvider()

    def generate(self, request: ExplanationRequest) -> ExplanationResponse:
        """
        Execute full explanation generation pipeline.

        Args:
            request: Formatted ExplanationRequest object.

        Returns:
            Validated ExplanationResponse instance.
        """
        # Step 1: Build system and user prompts
        system_prompt = build_system_prompt()
        user_prompt = build_user_prompt(request)

        # Step 2: Call LLM provider
        try:
            raw_response = self.provider.generate_explanation(system_prompt, user_prompt)
        except Exception as e:
            raw_response = json.dumps({
                "summary": "Unable to generate LLM explanation due to provider execution failure.",
                "recommendation_reasons": ["Provider error occurred."],
                "observed_user_patterns": [],
                "research_context": [],
                "limitations": [f"Provider execution error: {str(e)}"],
                "sources": []
            })

        # Step 3: Parse JSON response with malformed recovery
        try:
            # Clean markdown codeblocks if provider returns ```json ... ```
            clean_str = raw_response.strip()
            if clean_str.startswith("```json"):
                clean_str = clean_str[7:]
            if clean_str.startswith("```"):
                clean_str = clean_str[3:]
            if clean_str.endswith("```"):
                clean_str = clean_str[:-3]
            clean_str = clean_str.strip()

            res_dict = json.loads(clean_str)
        except Exception as e:
            res_dict = {
                "summary": "Malformed JSON response received from LLM provider.",
                "recommendation_reasons": ["JSON parsing failed."],
                "observed_user_patterns": [],
                "research_context": [],
                "limitations": [f"Malformed JSON parse error: {str(e)}"],
                "sources": []
            }

        # Step 4: Construct ExplanationResponse schema object
        response = ExplanationResponse.from_dict(res_dict)

        # Step 5: Execute grounding and safety validation
        validated_response = self.validator.validate(request, response)

        return validated_response
