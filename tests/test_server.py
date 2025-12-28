"""
Unit tests for external_models_mcp/server.py
"""
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

# Ensure the src directory is in the path for imports
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from external_models_mcp.server import (
    _resolve_model_name,
    _read_context_files,
    ask_expert,
    compare_experts,
    MODEL_ALIASES,
    MAX_FILE_SIZE_BYTES,
    MAX_TOTAL_CHARS
)


# --------------- _resolve_model_name tests --------------- #
@pytest.mark.parametrize("alias,expected", [
    ("glm", MODEL_ALIASES["glm"]),
    ("GLM", MODEL_ALIASES["glm"]),
    ("  glm  ", MODEL_ALIASES["glm"]),
    ("kimi", MODEL_ALIASES["kimi"]),
    ("kimi-k2", MODEL_ALIASES["kimi-k2"]),
    ("nonexistent", "nonexistent"),
])
def test_resolve_model_name(alias: str, expected: str):
    assert _resolve_model_name(alias) == expected


# --------------- _read_context_files tests --------------- #
def test_read_context_files_nonexistent_file():
    assert _read_context_files(["/nonexistent/path"]) == ""


def test_read_context_files_binary_file():
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(b"\x00\x01\x02")
        tmp.flush()
        content = _read_context_files([tmp.name])
        assert "<error" in content and "binary" in content
    os.unlink(tmp.name)


def test_read_context_files_large_file():
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as tmp:
        # Write content larger than MAX_FILE_SIZE_BYTES
        tmp.write("x" * (MAX_FILE_SIZE_BYTES + 1))
        tmp.flush()
        content = _read_context_files([tmp.name])
        assert "<error" in content and "too large" in content
    os.unlink(tmp.name)


def test_read_context_files_valid_content():
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".py") as tmp:
        tmp.write("print('hello world')")
        tmp.flush()
        content = _read_context_files([tmp.name])
        expected_path = Path(tmp.name).resolve()
        assert f"<file path='{expected_path}'>" in content
        assert "print('hello world')" in content
        assert "</file>" in content
    os.unlink(tmp.name)


def test_read_context_files_multiple_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        file1 = Path(tmpdir) / "a.py"
        file1.write_text("a = 1", encoding="utf-8")
        file2 = Path(tmpdir) / "b.py"
        file2.write_text("b = 2", encoding="utf-8")

        content = _read_context_files([str(file1), str(file2)])
        assert "a = 1" in content
        assert "b = 2" in content


def test_read_context_files_total_char_limit():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create files that together exceed MAX_TOTAL_CHARS
        file1 = Path(tmpdir) / "big.py"
        file1.write_text("x" * (MAX_TOTAL_CHARS // 2), encoding="utf-8")
        file2 = Path(tmpdir) / "bigger.py"
        file2.write_text("y" * (MAX_TOTAL_CHARS // 2 + 1000), encoding="utf-8")

        content = _read_context_files([str(file1), str(file2)])
        assert "Context limit reached" in content


# --------------- ask_expert tests --------------- #
@patch("external_models_mcp.server.litellm.completion")
def test_ask_expert_no_context(mock_completion):
    mock_completion.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="mocked response"))]
    )
    result = ask_expert("What is 2+2?")
    assert result == "mocked response"
    mock_completion.assert_called_once()
    call_args = mock_completion.call_args
    # Verify default model is kimi-k2
    assert call_args.kwargs["model"] == MODEL_ALIASES["kimi-k2"]
    messages = call_args.kwargs["messages"]
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "What is 2+2?"


@patch("external_models_mcp.server.litellm.completion")
def test_ask_expert_with_context(mock_completion):
    mock_completion.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="mocked response"))]
    )
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".py") as tmp:
        tmp.write("def hello(): pass")
        tmp.flush()
        result = ask_expert("Explain this code", context_files=[tmp.name])
        assert result == "mocked response"
        messages = mock_completion.call_args.kwargs["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert "def hello(): pass" in messages[0]["content"]
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Explain this code"
    os.unlink(tmp.name)


@patch("external_models_mcp.server.litellm.completion")
def test_ask_expert_model_alias(mock_completion):
    mock_completion.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="mocked response"))]
    )
    ask_expert("Q", model="glm")
    assert mock_completion.call_args.kwargs["model"] == MODEL_ALIASES["glm"]


@patch("external_models_mcp.server.litellm.completion")
def test_ask_expert_litellm_error(mock_completion):
    mock_completion.side_effect = Exception("litellm broke")
    result = ask_expert("Q")
    assert "Error using" in result and "litellm broke" in result


# --------------- compare_experts tests --------------- #
@patch("external_models_mcp.server.litellm.completion")
def test_compare_experts(mock_completion):
    def side_effect(*args, **kwargs):
        model = kwargs.get("model", "")
        if "moonshot" in model: # Kimi matches openrouter/moonshot...
            return MagicMock(choices=[MagicMock(message=MagicMock(content="kimi says"))])
        if "glm" in model: # GLM matches ...glm...
            return MagicMock(choices=[MagicMock(message=MagicMock(content="glm says"))])
        return MagicMock(choices=[MagicMock(message=MagicMock(content="other"))])

    mock_completion.side_effect = side_effect
    # Use actual aliases from the map to ensure mock catches them
    result = compare_experts("Compare", experts=["kimi", "glm"])
    assert "Expert: KIMI" in result
    assert "kimi says" in result
    assert "Expert: GLM" in result
    assert "glm says" in result
