"""
Builds the context payload handed to the LLM, with basic prompt-injection
mitigation on any text pulled from retrieved documents or user-supplied
context (both are attacker-influenced surfaces in a RAG system).
"""

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Patterns that are common in prompt-injection attempts embedded in documents.
# This is defense-in-depth, not a complete solution -- it should be paired
# with keeping the system prompt and instructions out of band from retrieved
# content wherever the LLM API supports separate system/user roles.
_INJECTION_PATTERNS = [
    re.compile(r"ignore (all |the )?previous instructions", re.IGNORECASE),
    re.compile(r"disregard (all |the )?(above|prior)", re.IGNORECASE),
    re.compile(r"you are now\b", re.IGNORECASE),
    re.compile(r"system prompt", re.IGNORECASE),
    re.compile(r"reveal your (instructions|prompt)", re.IGNORECASE),
]


class ContextBuilder:
    def __init__(self, config: Dict[str, Any]):
        self.max_context_chars = config.get("max_context_chars", 8000)
        self.max_docs_in_context = config.get("max_docs_in_context", 5)
        self.sanitize = config.get("sanitize_inputs", True)

    def build(self, query: str, docs: List[Dict], user_context: Optional[Dict] = None) -> Dict[str, Any]:
        context = {
            "query": self._sanitize_text(query),
            "documents": [self._sanitize_doc(d) for d in docs[: self.max_docs_in_context]],
            "user_context": self._sanitize_dict(user_context or {}),
        }
        context["text"] = self._render(context)
        return context

    def _sanitize_text(self, text: str) -> str:
        if not text:
            return ""
        if self.sanitize:
            for pattern in _INJECTION_PATTERNS:
                text = pattern.sub("[REDACTED]", text)
        return text[: self.max_context_chars]

    def _sanitize_doc(self, doc: Dict) -> Dict:
        return {
            "id": doc.get("id", ""),
            "text": self._sanitize_text(doc.get("text", "")),
            "score": doc.get("score", 0.0),
        }

    def _sanitize_dict(self, d: Dict) -> Dict:
        out = {}
        for k, v in d.items():
            if isinstance(v, str):
                out[k] = self._sanitize_text(v)
            elif isinstance(v, dict):
                out[k] = self._sanitize_dict(v)
            else:
                out[k] = v
        return out

    @staticmethod
    def _render(context: Dict) -> str:
        lines = [f"Query: {context['query']}", "", "Retrieved documents:"]
        for i, doc in enumerate(context["documents"], start=1):
            lines.append(f"[{i}] {doc['text']}")
        return "\n".join(lines)
