from unittest.mock import patch

import pytest

from src.deriver.prompts import (
    estimate_deriver_prompt_tokens,
    estimate_minimal_deriver_prompt_tokens,
    minimal_deriver_prompt,
)
from src.utils.representation import PromptRepresentation


def test_minimal_deriver_prompt_includes_custom_instructions_when_present() -> None:
    prompt = minimal_deriver_prompt(
        peer_id="alice",
        messages="alice: hello",
        custom_instructions="Prefer concrete timeline facts.",
    )

    assert "CUSTOM INSTRUCTIONS:" in prompt
    assert "Prefer concrete timeline facts." in prompt


def test_minimal_deriver_prompt_omits_custom_instructions_when_absent() -> None:
    prompt = minimal_deriver_prompt(
        peer_id="alice",
        messages="alice: hello",
        custom_instructions=None,
    )

    assert "CUSTOM INSTRUCTIONS:" not in prompt


def test_minimal_deriver_prompt_uses_json_examples_without_type_labels() -> None:
    prompt = minimal_deriver_prompt(
        peer_id="alice",
        messages="alice: hello",
        custom_instructions=None,
    )

    assert "[EXPLICIT]" not in prompt
    assert "- EXPLICIT:" not in prompt
    assert "EXPLICIT:" not in prompt
    assert '{"explicit":[{"content":"alice is 25 years old"}]}' in prompt
    assert "<examples>" in prompt
    assert "</examples>" in prompt


def test_prompt_representation_schema_has_no_biography_examples() -> None:
    blob = str(PromptRepresentation.model_json_schema())
    assert "Rover" not in blob
    assert "The user is 25" not in blob
    assert "[EXPLICIT]" not in blob
    assert "type-label" in blob


def test_estimate_deriver_prompt_tokens_increases_with_custom_instructions() -> None:
    base_tokens = estimate_minimal_deriver_prompt_tokens()
    custom_tokens = estimate_deriver_prompt_tokens(
        "Prefer explicit facts with absolute dates and keep the subject precise."
    )

    assert custom_tokens > base_tokens


def test_estimate_deriver_prompt_tokens_propagates_token_estimation_errors() -> None:
    estimate_minimal_deriver_prompt_tokens.cache_clear()

    with patch(
        "src.deriver.prompts.estimate_tokens",
        side_effect=RuntimeError("tokenizer unavailable"),
    ):
        with pytest.raises(RuntimeError, match="tokenizer unavailable"):
            estimate_deriver_prompt_tokens(None)

        with pytest.raises(RuntimeError, match="tokenizer unavailable"):
            estimate_deriver_prompt_tokens("Prefer concrete facts.")
