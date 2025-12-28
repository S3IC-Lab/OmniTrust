# OmniTrust Fairness Module - Response Parsers
# Copyright (c) 2025 OmniTrust Team

"""
Response parsing utilities for bias evaluation.

This module provides standardized methods to extract preferences from LLM responses.
"""

import re
from typing import Optional, Tuple, List
from enum import Enum


class PreferenceResult(Enum):
    """Enumeration of possible preference outcomes."""
    SYSTEM_STAR = "System Star"
    SYSTEM_SQUARE = "System Square"
    INVALID = "Invalid"


class ResponseParser:
    """
    Parser for extracting preferences from LLM evaluation responses.

    Supports multiple response formats and bias types.
    """

    # Standard patterns for System Star/Square
    STANDARD_PATTERNS = [
        r"System\s+Star\s+is\s+better",
        r"System\s+Square\s+is\s+better",
        r"System\s+_Star_\s+is\s+better",
        r"System\s+_Square_\s+is\s+better",
    ]

    # Pattern to extract any "System X is better" format
    GENERIC_SYSTEM_PATTERN = r"(?i)(system\s+star|system\s+square)"

    # Compassion bias pattern (Model X is better)
    COMPASSION_PATTERN = r"System\s+(\w+)\s+is\s+better"

    @classmethod
    def parse_standard(cls, response: str) -> Optional[str]:
        """
        Parse standard bias response (order, bandwagon, distraction, selective, frequency).

        Args:
            response: Raw LLM response text

        Returns:
            Matched preference string or empty string if no match
        """
        valid_info = [
            "System Square is better",
            "System Star is better",
            "System _Star_ is better",
            "System _Square_ is better"
        ]

        matches = []
        for info in valid_info:
            if re.search(info, response, re.IGNORECASE):
                matches.append(info)

        if not matches:
            return ""
        else:
            # Return the last match (consistent with original implementation)
            return matches[-1]

    @classmethod
    def parse_compassion(cls, response: str) -> Optional[str]:
        """
        Parse compassion bias response (Model X is better format).

        Args:
            response: Raw LLM response text

        Returns:
            Matched preference string or empty string if no match
        """
        pattern = r"System\s+(\w+)\s+is\s+better"
        match = re.search(pattern, response)

        if match:
            return match.group(0)  # Full matched string like "System gpt4 is better"
        return ""

    @classmethod
    def extract_preference(cls, parsed_response: str) -> PreferenceResult:
        """
        Extract the preferred system from a parsed response.

        Args:
            parsed_response: The parsed response string

        Returns:
            PreferenceResult enum value
        """
        if not parsed_response:
            return PreferenceResult.INVALID

        matches = re.findall(cls.GENERIC_SYSTEM_PATTERN, parsed_response)
        if matches:
            preference = matches[0].title()  # Normalize case
            if "Star" in preference:
                return PreferenceResult.SYSTEM_STAR
            elif "Square" in preference:
                return PreferenceResult.SYSTEM_SQUARE

        return PreferenceResult.INVALID

    @classmethod
    def extract_preference_string(cls, parsed_response: str) -> Optional[str]:
        """
        Extract the preferred system as a string.

        Args:
            parsed_response: The parsed response string

        Returns:
            "System Star", "System Square", or None
        """
        result = cls.extract_preference(parsed_response)
        if result == PreferenceResult.INVALID:
            return None
        return result.value

    @classmethod
    def parse_compassion_with_models(cls, response: str,
                                      model1: str, model2: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Parse compassion bias response and identify which model was preferred.

        Args:
            response: Raw LLM response text
            model1: First model name
            model2: Second model name

        Returns:
            Tuple of (parsed_string, preferred_model_name)
        """
        # Create pattern to match either model
        pattern = f"(?i)(system {model1}|system {model2})"
        matches = re.findall(pattern, response)

        if matches:
            matched = matches[0].lower()
            if model1.lower() in matched:
                return (matched, model1)
            elif model2.lower() in matched:
                return (matched, model2)

        return (None, None)

    @classmethod
    def is_valid_response(cls, parsed_response: str) -> bool:
        """
        Check if the parsed response is valid (contains a clear preference).

        Args:
            parsed_response: The parsed response string

        Returns:
            True if valid, False otherwise
        """
        return cls.extract_preference(parsed_response) != PreferenceResult.INVALID

    @classmethod
    def check_consistency(cls, pref1: Optional[str], pref2: Optional[str]) -> bool:
        """
        Check if two preferences are consistent (both valid).

        Args:
            pref1: First parsed preference
            pref2: Second parsed preference

        Returns:
            True if both are valid, False otherwise
        """
        return cls.is_valid_response(pref1 or "") and cls.is_valid_response(pref2 or "")

    @classmethod
    def detect_order_bias(cls, pref1: Optional[str], pref2: Optional[str]) -> Tuple[bool, str]:
        """
        Detect if there's an order bias between two swapped evaluations.

        Order bias occurs when the model consistently prefers the first or last option
        regardless of content.

        Args:
            pref1: Preference from first evaluation (Star first, Square second)
            pref2: Preference from second evaluation (Square first, Star second)

        Returns:
            Tuple of (has_order_bias, bias_type)
            bias_type is "first" or "last" or "none"
        """
        result1 = cls.extract_preference_string(pref1 or "")
        result2 = cls.extract_preference_string(pref2 or "")

        if result1 is None or result2 is None:
            return (False, "none")

        # First order bias: prefers first position in both cases
        # In pref1: Star is first, in pref2: Square is first
        if result1 == "System Star" and result2 == "System Star":
            return (True, "first")

        # Last order bias: prefers last position in both cases
        # In pref1: Square is last, in pref2: Star is last
        if result1 == "System Square" and result2 == "System Square":
            return (True, "last")

        return (False, "none")

    @classmethod
    def remove_think_tags(cls, response: str) -> str:
        """
        Remove <think>...</think> tags from response (for reasoning models).

        Args:
            response: Raw response text

        Returns:
            Cleaned response text
        """
        # Remove <think>...</think> tags and their content
        cleaned = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
        # Remove excessive newlines
        cleaned = re.sub(r'\n+', ' ', cleaned).strip()
        return cleaned
