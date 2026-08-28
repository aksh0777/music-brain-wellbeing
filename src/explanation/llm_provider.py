"""
Modular LLM Provider Abstraction Module

WHAT: Provides an abstract `LLMProvider` interface and concrete implementations (`DemoLLMProvider` and `GenericOpenAILLMProvider`).

WHY: Decouples application logic from specific LLM vendors (OpenAI, Groq, Ollama, local models).
Allows 100% offline, deterministic execution in DEMO mode and unit tests without external API dependencies.

PROVIDER ARCHITECTURE:
    LLMProvider (Abstract Base Class)
       ├── DemoLLMProvider (Offline Deterministic Mock for Tests & Notebooks)
       └── GenericOpenAILLMProvider (OpenAI-Compatible REST Client configured via env vars)
"""

from abc import ABC, abstractmethod
import os
import json
import requests
from typing import Dict, Any, Optional


class LLMProvider(ABC):
    """
    Abstract interface for LLM text generation providers.
    """

    @abstractmethod
    def generate_explanation(self, system_prompt: str, user_prompt: str) -> str:
        """
        Generate raw text response string given system and user prompts.

        Args:
            system_prompt: High-level instructions and safety constraints.
            user_prompt: Structured JSON/text request payload.

        Returns:
            Raw LLM text response string (expected to be JSON format).
        """
        pass


class DemoLLMProvider(LLMProvider):
    """
    Deterministic, offline mock LLM provider for unit tests and local demonstration notebooks.
    Requires no API keys or network access.
    """

    def generate_explanation(self, system_prompt: str, user_prompt: str) -> str:
        """
        Synthesize a realistic, structured JSON explanation using information parsed from user_prompt.
        """
        try:
            prompt_data = json.loads(user_prompt)
        except Exception:
            prompt_data = {}

        user_profile = prompt_data.get("user_profile", {})
        recs = prompt_data.get("recommendations", [])
        evidence = prompt_data.get("evidence_package", {})

        # Extract track names / IDs for deterministic reasons
        rec_reasons = []
        for track in recs[:3]:
            track_id = track.get("track_id", "TRK_UNKNOWN")
            track_name = track.get("track_name") or track.get("name") or "Recommended Track"
            sim_score = track.get("similarity_score", 0.85)
            rec_reasons.append(
                f"Track '{track_name}' (ID: {track_id}) matches your preference for "
                f"soothing acoustics with similarity score {sim_score:.2f}."
            )

        # Extract research context from evidence package
        research_context = []
        sources = []

        sources_data = evidence.get("sources", [])
        retrieved_chunks = evidence.get("retrieved_chunks", [])

        if sources_data:
            for src in sources_data[:2]:
                title = src.get("title", "Research Study")
                pmid = src.get("pmid", "")
                doi = src.get("doi", "")
                year = src.get("year", 2021)
                sources.append({
                    "title": title,
                    "pmid": pmid,
                    "doi": doi,
                    "year": year
                })
                research_context.append(
                    f"Research by {src.get('authors', 'authors')} ({year}, PMID: {pmid}) "
                    f"investigated associations between music interventions and stress recovery."
                )

        if not research_context:
            research_context.append("No specific peer-reviewed research evidence was retrieved for this acoustic profile.")

        summary = (
            f"Recommended {len(recs)} tracks aligned with your acoustic preference profile. "
            "Retrieved scientific literature provides contextual observational evidence regarding these acoustic traits."
        )

        observed_patterns = [
            f"User profile indicates average session length of {user_profile.get('avg_tracks_per_session', 5.0)} tracks.",
            f"Listening history displays preference for acoustic features (data provenance: {user_profile.get('data_provenance', 'demo')})."
        ]

        limitations = [
            "This system provides acoustic preference matching and scientific context, NOT a clinical diagnosis or treatment.",
            "Retrieved literature shows context-dependent findings; listening to recommendations does not guarantee physiological anxiety reduction."
        ]

        response_dict = {
            "summary": summary,
            "recommendation_reasons": rec_reasons,
            "observed_user_patterns": observed_patterns,
            "research_context": research_context,
            "limitations": limitations,
            "sources": sources
        }

        return json.dumps(response_dict, indent=2)


class GenericOpenAILLMProvider(LLMProvider):
    """
    OpenAI-compatible REST client reading API credentials exclusively from environment variables.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        model_name: Optional[str] = None
    ):
        self.api_key = api_key or os.getenv("LLM_API_KEY", "")
        self.api_base = api_base or os.getenv("LLM_API_BASE", "https://api.openai.com/v1")
        self.model_name = model_name or os.getenv("LLM_MODEL_NAME", "gpt-4o-mini")

        if self.api_base.endswith("/"):
            self.api_base = self.api_base[:-1]

    def generate_explanation(self, system_prompt: str, user_prompt: str) -> str:
        """
        Send HTTP POST request to OpenAI-compatible `/chat/completions` API endpoint.
        """
        if not self.api_key:
            raise ValueError("LLM_API_KEY environment variable is missing. Set credentials or use DemoLLMProvider.")

        url = f"{self.api_base}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.2
        }

        response = requests.post(url, headers=headers, json=payload, timeout=30)
        if response.status_code != 200:
            raise RuntimeError(f"LLM API Error (Status {response.status_code}): {response.text}")

        res_json = response.json()
        content = res_json["choices"][0]["message"]["content"]
        return content
