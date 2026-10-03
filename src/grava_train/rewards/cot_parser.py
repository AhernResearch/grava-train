"""Decision labels and reasoning sections used by the registered rewards."""

import re
from typing import Optional

_DECISION_ALIASES = {
    "follow": "Follow",
    "stop": "Stop",
    "yield": "Yield",
    "nudge left": "Nudge Left",
    "nudgeleft": "Nudge Left",
    "nudge_left": "Nudge Left",
    "nudge right": "Nudge Right",
    "nudgeright": "Nudge Right",
    "nudge_right": "Nudge Right",
    "overtake": "Overtake",
    "pass": "Overtake",
    "caution": "Caution",
    "monitor": "Caution",
}



def extract_decision(text: str) -> str | None:
    """Extract a decision from the final answer, preserving exact/soft matching."""
    source = extract_answer_strict(text)
    if source is None:
        source = text
    match = re.search(r"decided:\s*(.+?)\.?\s*$", source, re.IGNORECASE | re.MULTILINE)
    if match:
        return _normalize_decision(match.group(1).strip())
    return _match_decision_in_text(source)


def _normalize_decision(raw: str) -> Optional[str]:
    """Normalize raw decision text to canonical label."""
    key = raw.lower().strip().rstrip(".")
    if key in _DECISION_ALIASES:
        return _DECISION_ALIASES[key]
    # Fuzzy: check if any canonical label is a substring
    for alias, label in _DECISION_ALIASES.items():
        if alias in key:
            return label
    return None


def _match_decision_in_text(text: str) -> Optional[str]:
    """Try to find a decision label directly in text."""
    text_lower = text.lower()
    # Check from most specific (longest) to least specific
    for label in ["nudge left", "nudge right", "overtake", "caution", "follow", "stop", "yield"]:
        if label in text_lower:
            return _DECISION_ALIASES[label]
    return None


def detect_reasoning_mode(text: str) -> str:
    """Detect reasoning mode from model output.

    Returns "think", "chain", or "none".
    When both <think> and <chain> tags are present, returns "think"
    (think is the outer container that may include chain DSL).
    """
    has_think = "<think>" in text and "</think>" in text
    has_chain = "<chain>" in text and "</chain>" in text
    if has_think:
        return "think"
    if has_chain:
        return "chain"
    return "none"


def extract_answer_strict(text: str) -> Optional[str]:
    """Extract answer content from model output.

    Supports both old format (<answer>...</answer>) and
    new Qwen3.5 format (content after </think>).
    """
    # Old format: explicit <answer> tag
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # New format: content after </think>
    m = re.search(r"</think>\s*\n?\n?(.*)", text, re.DOTALL | re.IGNORECASE)
    if m:
        content = m.group(1).strip()
        if content:
            return content
    return None
