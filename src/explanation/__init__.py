"""
Grounded Non-Clinical LLM Explanation Layer Package

WHAT: Provides structured explanation requests/responses, prompt engineering, modular LLM abstraction,
grounding validation, and pipeline coordination for the Music Brain Wellbeing System.
"""

from .schemas import SafetyConstraints, ExplanationRequest, ExplanationResponse
from .llm_provider import LLMProvider, DemoLLMProvider, GenericOpenAILLMProvider
from .prompt import build_system_prompt, build_user_prompt
from .validation import GroundingValidator
from .explanation_generator import ExplanationGenerator

__all__ = [
    "SafetyConstraints",
    "ExplanationRequest",
    "ExplanationResponse",
    "LLMProvider",
    "DemoLLMProvider",
    "GenericOpenAILLMProvider",
    "build_system_prompt",
    "build_user_prompt",
    "GroundingValidator",
    "ExplanationGenerator",
]
