from unittest.mock import MagicMock, patch

import pytest
from langdetect import LangDetectException

from fedotllm.agents.translator import TranslatorAgent

from fedotllm.llm import AIInference

@pytest.fixture
def mock_inference_fixture():
    return MagicMock(spec=AIInference)


class TestTranslatorAgent:
    @patch("fedotllm.agents.translator.detect")
    def test_language_detection_success(self, mock_detect, mock_inference_fixture):
        mock_detect.return_value = "es"
        agent = TranslatorAgent(inference=mock_inference_fixture)
        mock_inference_fixture.create.return_value = "Translated text"
        agent.translate_input_to_english("Hola mundo")
        assert agent.source_language == "es"
        mock_detect.assert_called_once_with("Hola mundo")

    @patch("fedotllm.agents.translator.detect")
    def test_language_detection_failure_defaults_to_english(
        self, mock_detect, mock_inference_fixture
    ):
        mock_detect.side_effect = LangDetectException(0, "Detection failed")
        agent = TranslatorAgent(inference=mock_inference_fixture)
        mock_inference_fixture.create.return_value = "Translated text"
        agent.translate_input_to_english("Invalid text for detection")
        assert agent.source_language == "en"
        mock_detect.assert_called_once_with("Invalid text for detection")

    @patch("fedotllm.agents.translator.detect")
    def test_translation_to_english_success(self, mock_detect, mock_inference_fixture):
        mock_detect.return_value = "es"
        mock_inference_fixture.query.return_value = "Hello world"

        agent = TranslatorAgent(inference=mock_inference_fixture)
        translated_text = agent.translate_input_to_english("Hola mundo")

        assert translated_text == "Hello world"
        assert agent.source_language == "es"
        mock_inference_fixture.query.assert_called_once()

        args, kwargs = mock_inference_fixture.query.call_args
        prompt = args[0] if args else kwargs.get("messages", "")
        assert "Translate the following text from es to en." in prompt
        assert "Hola mundo" in prompt

    @patch("fedotllm.agents.translator.detect")
    def test_translation_to_english_inference_error_returns_original(
        self, mock_detect, mock_inference_fixture
    ):
        mock_detect.return_value = "es"
        mock_inference_fixture.query.side_effect = Exception("LLM API error")

        agent = TranslatorAgent(inference=mock_inference_fixture)
        original_text = "Hola mundo"
        translated_text = agent.translate_input_to_english(original_text)

        assert translated_text == original_text
        assert agent.source_language == "es"

    @patch("fedotllm.agents.translator.detect")
    def test_english_input_not_translated(self, mock_detect, mock_inference_fixture):
        mock_detect.return_value = "en"

        agent = TranslatorAgent(inference=mock_inference_fixture)
        input_text = "Hello world, this is English."
        translated_text = agent.translate_input_to_english(input_text)

        assert translated_text == input_text
        assert agent.source_language == "en"
        mock_inference_fixture.query.assert_not_called()

    @patch("fedotllm.agents.translator.detect")
    def test_translation_to_source_language_success(
        self, mock_detect, mock_inference_fixture
    ):
        mock_detect.return_value = "es"

        mock_inference_fixture.query.return_value = "Hello world"
        agent = TranslatorAgent(inference=mock_inference_fixture)
        agent.translate_input_to_english("Hola mundo")
        assert agent.source_language == "es"

        mock_inference_fixture.query.reset_mock()
        mock_inference_fixture.query.return_value = "Hola mundo otra vez"

        translated_output = agent.translate_output_to_source_language(
            "Hello world again"
        )

        assert translated_output == "Hola mundo otra vez"
        mock_inference_fixture.query.assert_called_once()
        args, kwargs = mock_inference_fixture.query.call_args
        prompt = args[0] if args else kwargs.get("messages", "")
        assert "Translate the following text from English to es." in prompt
        assert "Hello world again" in prompt

    def test_extract_code_blocks(self, mock_inference_fixture):
        agent = TranslatorAgent(inference=mock_inference_fixture)
        text_with_code = """Some text
```python
print('hello')
```
More text
```
x = 1
```"""
        processed_text, code_map = agent._extract_code_blocks(text_with_code)

        placeholder_0 = f"{agent.code_block_placeholder_prefix}_0__"
        placeholder_1 = f"{agent.code_block_placeholder_prefix}_1__"

        assert placeholder_0 in processed_text
        assert placeholder_1 in processed_text
        assert "print('hello')" not in processed_text
        assert "x = 1" not in processed_text

        assert len(code_map) == 2
        assert code_map[placeholder_0] == "```python\nprint('hello')\n```"
        assert code_map[placeholder_1] == "```\nx = 1\n```"

    def test_extract_code_blocks_empty_string(self, mock_inference_fixture):
        agent = TranslatorAgent(inference=mock_inference_fixture)
        processed_text, code_map = agent._extract_code_blocks("")
        assert processed_text == ""
        assert not code_map

    def test_extract_code_blocks_no_code_blocks(self, mock_inference_fixture):
        agent = TranslatorAgent(inference=mock_inference_fixture)
        text = "This is a simple text without any code blocks."
        processed_text, code_map = agent._extract_code_blocks(text)
        assert processed_text == text
        assert not code_map

    def test_extract_code_blocks_entirely_code_block(self, mock_inference_fixture):
        agent = TranslatorAgent(inference=mock_inference_fixture)
        text = "```python\nprint('entirely code')\n```"
        processed_text, code_map = agent._extract_code_blocks(text)
        placeholder = f"{agent.code_block_placeholder_prefix}_0__"
        assert processed_text == placeholder
        assert len(code_map) == 1
        assert code_map[placeholder] == text

    def test_extract_code_blocks_adjacent_blocks(self, mock_inference_fixture):
        agent = TranslatorAgent(inference=mock_inference_fixture)
        text = "```python\nprint('block1')\n``````text\nblock2\n```"
        processed_text, code_map = agent._extract_code_blocks(text)
        placeholder0 = f"{agent.code_block_placeholder_prefix}_0__"
        placeholder1 = f"{agent.code_block_placeholder_prefix}_1__"
        assert processed_text == f"{placeholder0}{placeholder1}"
        assert len(code_map) == 2
        assert code_map[placeholder0] == "```python\nprint('block1')\n```"
        assert code_map[placeholder1] == "```text\nblock2\n```"

    def test_reinsert_code_blocks(self, mock_inference_fixture):
        agent = TranslatorAgent(inference=mock_inference_fixture)
        text_with_placeholders = f"""Texto traducido
{agent.code_block_placeholder_prefix}_0__
Más texto traducido
{agent.code_block_placeholder_prefix}_1__"""
        code_map = {
            f"{agent.code_block_placeholder_prefix}_0__": "```python\nprint('hello')\n```",
            f"{agent.code_block_placeholder_prefix}_1__": "```\nx = 1\n```",
        }

        final_text = agent._reinsert_code_blocks(text_with_placeholders, code_map)

        expected_text = """Texto traducido
```python
print('hello')
```
Más texto traducido
```
x = 1
```"""
        assert final_text == expected_text
        assert agent.code_block_placeholder_prefix not in final_text

    def test_reinsert_code_blocks_empty_map(self, mock_inference_fixture):
        agent = TranslatorAgent(inference=mock_inference_fixture)
        text = f"Text with {agent.code_block_placeholder_prefix}_0__ placeholder"
        final_text = agent._reinsert_code_blocks(text, {})
        assert final_text == text  # Placeholder should remain if not in map

    def test_reinsert_code_blocks_placeholder_missing_from_map(
        self, mock_inference_fixture
    ):
        agent = TranslatorAgent(inference=mock_inference_fixture)
        placeholder = f"{agent.code_block_placeholder_prefix}_0__"
        text = f"Text with {placeholder} and {agent.code_block_placeholder_prefix}_1__"
        code_map = {f"{agent.code_block_placeholder_prefix}_1__": "```\ncode1\n```"}
        # Expect placeholder_0 to remain as is, placeholder_1 to be replaced
        expected_text = f"Text with {placeholder} and ```\ncode1\n```"
        final_text = agent._reinsert_code_blocks(text, code_map)
        assert final_text == expected_text

    @patch("fedotllm.agents.translator.logger.warning")
    def test_reinsert_code_blocks_placeholder_not_in_text(
        self, mock_log_warning, mock_inference_fixture
    ):
        agent = TranslatorAgent(inference=mock_inference_fixture)
        text = "Translated text without any placeholders."
        code_map = {
            f"{agent.code_block_placeholder_prefix}_0__": "```\ncode0\n```",
            f"{agent.code_block_placeholder_prefix}_1__": "```\ncode1\n```",
        }
        final_text = agent._reinsert_code_blocks(text, code_map)
        assert final_text == text
        # Check that warnings were logged for unused placeholders
        assert (
            mock_log_warning.call_count == 3
        )  # Two for individual placeholders, one summary
        mock_log_warning.assert_any_call(
            f"_reinsert_code_blocks: Placeholder '{agent.code_block_placeholder_prefix}_0__' not found in the translated text for re-insertion."
        )
        mock_log_warning.assert_any_call(
            f"_reinsert_code_blocks: Placeholder '{agent.code_block_placeholder_prefix}_1__' not found in the translated text for re-insertion."
        )
        mock_log_warning.assert_any_call(
            f"_reinsert_code_blocks: Reinserted 0 out of 2 code blocks from map. "
            f"Placeholders not found in text: ['{agent.code_block_placeholder_prefix}_0__', '{agent.code_block_placeholder_prefix}_1__']"
        )

    @patch.object(TranslatorAgent, "_reinsert_code_blocks")
    @patch.object(TranslatorAgent, "_extract_code_blocks")
    def test_translate_text_source_language_none(
        self, mock_extract, mock_reinsert, mock_inference_fixture
    ):
        agent = TranslatorAgent(inference=mock_inference_fixture)
        text = "Some text"
        processed_text = "Processed text"
        llm_response = "Translated text"
        final_text = "Final text"

        mock_extract.return_value = (processed_text, {"placeholder": "code"})
        mock_inference_fixture.query.return_value = llm_response
        mock_reinsert.return_value = final_text

        result = agent._translate_text(text, "en", source_language=None)

        assert result == final_text
        mock_inference_fixture.query.assert_called_once()
        prompt = mock_inference_fixture.query.call_args[0][0]
        assert (
            "Translate the following text from the auto-detected source language to en."
            in prompt
        )
        assert processed_text in prompt
        mock_extract.assert_called_once_with(text)
        mock_reinsert.assert_called_once_with(llm_response, {"placeholder": "code"})

    @patch.object(TranslatorAgent, "_reinsert_code_blocks")
    @patch.object(TranslatorAgent, "_extract_code_blocks")
    def test_translate_text_source_language_en(
        self, mock_extract, mock_reinsert, mock_inference_fixture
    ):
        agent = TranslatorAgent(inference=mock_inference_fixture)
        text = "Some text"
        mock_extract.return_value = (text, {})  # No code blocks
        mock_inference_fixture.query.return_value = "Translated text"
        mock_reinsert.return_value = "Translated text"  # Result from reinsert

        agent._translate_text(text, "fr", source_language="en")

        mock_inference_fixture.query.assert_called_once()
        prompt = mock_inference_fixture.query.call_args[0][0]
        assert "Translate the following text from English to fr." in prompt
        assert text in prompt
        mock_extract.assert_called_once_with(text)
        mock_reinsert.assert_called_once_with("Translated text", {})

    @patch("fedotllm.agents.translator.logger.warning")
    @patch.object(TranslatorAgent, "_extract_code_blocks")
    def test_translate_text_llm_returns_none(
        self, mock_extract, mock_log_warning, mock_inference_fixture
    ):
        agent = TranslatorAgent(inference=mock_inference_fixture)
        text = "Some text with ```code```"
        processed_text = "Some text with placeholder"
        mock_extract.return_value = (processed_text, {"placeholder": "```code```"})
        mock_inference_fixture.query.return_value = None  # LLM returns None

        result = agent._translate_text(text, "es", "fr")

        assert result == text  # Should return original text
        mock_log_warning.assert_called_once_with(
            "LLM query did not return a valid string (type: <class 'NoneType'>, value: 'None...'). Returning original text."
        )
        mock_extract.assert_called_once_with(text)
        # _reinsert_code_blocks should not be called as _translate_text should return early

    @patch("fedotllm.agents.translator.logger.warning")
    @patch.object(TranslatorAgent, "_extract_code_blocks")
    def test_translate_text_llm_returns_non_string(
        self, mock_extract, mock_log_warning, mock_inference_fixture
    ):
        agent = TranslatorAgent(inference=mock_inference_fixture)
        text = "Some text"
        processed_text = (
            "Processed text"  # This won't actually be used if LLM fails early
        )
        mock_extract.return_value = (
            processed_text,
            {},
        )  # Mocked but _translate_text might return before using its output fully
        mock_inference_fixture.query.return_value = 123  # LLM returns an int

        result = agent._translate_text(text, "es", "fr")

        assert result == text  # Should return original text due to code change
        mock_log_warning.assert_called_once_with(
            "LLM query did not return a valid string (type: <class 'int'>, value: '123...'). Returning original text."
        )
        mock_extract.assert_called_once_with(
            text
        )  # extract is still called before the query

    @patch.object(TranslatorAgent, "_reinsert_code_blocks")
    @patch.object(TranslatorAgent, "_extract_code_blocks")
    def test_translate_text_processed_equals_original(
        self, mock_extract, mock_reinsert, mock_inference_fixture
    ):
        agent = TranslatorAgent(inference=mock_inference_fixture)
        text = "Text without code blocks."
        llm_response = "Translated text without code blocks."

        mock_extract.return_value = (
            text,
            {},
        )  # processed_text is the same as text, empty code_map
        mock_inference_fixture.query.return_value = llm_response
        mock_reinsert.return_value = (
            llm_response  # reinsert will just return llm_response
        )

        result = agent._translate_text(text, "es", "fr")

        assert result == llm_response
        mock_inference_fixture.query.assert_called_once()
        prompt = mock_inference_fixture.query.call_args[0][0]
        assert text in prompt  # Original text goes to LLM
        mock_extract.assert_called_once_with(text)
        mock_reinsert.assert_called_once_with(llm_response, {})

    @patch.object(TranslatorAgent, "_reinsert_code_blocks")
    @patch.object(TranslatorAgent, "_extract_code_blocks")
    def test_translate_text_llm_returns_same_as_processed(
        self, mock_extract, mock_reinsert, mock_inference_fixture
    ):
        agent = TranslatorAgent(inference=mock_inference_fixture)
        text = "Text with ```code```"
        processed_text = "Text with placeholder"  # Text after code extraction
        code_map = {"placeholder": "```code```"}

        mock_extract.return_value = (processed_text, code_map)
        # LLM returns the same text it received (the one with placeholders)
        mock_inference_fixture.query.return_value = processed_text
        # _reinsert_code_blocks will put the original code back
        mock_reinsert.side_effect = lambda txt, cmap: txt.replace(
            "placeholder", cmap["placeholder"]
        )

        result = agent._translate_text(text, "es", "fr")

        assert result == text  # Expect original text after reinsertion
        mock_inference_fixture.query.assert_called_once()
        prompt = mock_inference_fixture.query.call_args[0][0]
        assert processed_text in prompt
        mock_extract.assert_called_once_with(text)
        mock_reinsert.assert_called_once_with(processed_text, code_map)

    @patch("fedotllm.agents.translator.detect")
    def test_translate_input_to_english_empty_message(
        self, mock_detect, mock_inference_fixture
    ):
        agent = TranslatorAgent(inference=mock_inference_fixture)
        translated_text = agent.translate_input_to_english("")
        assert translated_text == ""
        mock_detect.assert_not_called()
        mock_inference_fixture.query.assert_not_called()
        assert agent.source_language is None  # Should not be set

    @patch.object(TranslatorAgent, "_translate_text")
    def test_translate_output_to_source_language_empty_message(
        self, mock_translate_text, mock_inference_fixture
    ):
        agent = TranslatorAgent(inference=mock_inference_fixture)
        agent.source_language = "es"  # Assume a source language was set
        translated_output = agent.translate_output_to_source_language("")
        assert translated_output == ""
        mock_translate_text.assert_not_called()

    @patch.object(TranslatorAgent, "_translate_text")
    def test_translate_output_to_source_language_no_source_lang(
        self, mock_translate_text, mock_inference_fixture
    ):
        agent = TranslatorAgent(inference=mock_inference_fixture)
        agent.source_language = None  # Source language not detected/set
        message = "Some output message"
        translated_output = agent.translate_output_to_source_language(message)
        assert (
            translated_output == message
        )  # Should return original if source_lang is None
        mock_translate_text.assert_not_called()

    @patch.object(TranslatorAgent, "_translate_text")
    def test_translate_output_to_source_language_is_english(
        self, mock_translate_text, mock_inference_fixture
    ):
        agent = TranslatorAgent(inference=mock_inference_fixture)
        agent.source_language = "en"  # Source language was English
        message = "Some output message"
        translated_output = agent.translate_output_to_source_language(message)
        assert (
            translated_output == message
        )  # Should return original if source_lang is 'en'
        mock_translate_text.assert_not_called()

    @patch("fedotllm.agents.translator.detect")
    def test_code_block_preservation_e2e(self, mock_detect, mock_inference_fixture):
        mock_detect.return_value = "es"
        original_text = """Texto antes
```python
# Esto es un comentario
print('Hola')
```
Texto después"""

        # Use the agent's placeholder prefix to construct the expected placeholder
        agent_for_placeholder = TranslatorAgent(
            inference=mock_inference_fixture
        )  # Temp agent to get prefix if needed
        placeholder_0 = f"{agent_for_placeholder.code_block_placeholder_prefix}_0__"

        placeholder_text_from_llm = f"""Translated before
{placeholder_0}
Translated after"""
        mock_inference_fixture.query.return_value = placeholder_text_from_llm

        agent = TranslatorAgent(inference=mock_inference_fixture)
        translated_text = agent.translate_input_to_english(original_text)

        expected_final_text = """Translated before
```python
# Esto es un comentario
print('Hola')
```
Translated after"""
        assert translated_text == expected_final_text

        mock_inference_fixture.query.assert_called_once()
        args, kwargs = mock_inference_fixture.query.call_args
        prompt = args[0] if args else kwargs.get("messages", "")
        assert (
            f"placeholders like '{agent.code_block_placeholder_prefix}_NUMBER__'"
            in prompt
        )
        assert "MUST NOT be translated or altered" in prompt
        assert placeholder_0 in prompt

    @patch("fedotllm.agents.translator.detect")
    def test_markdown_preservation_prompting(self, mock_detect, mock_inference_fixture):
        mock_detect.return_value = "fr"
        original_text = "# Titre\nCeci est du **gras** et de l'*italique*."

        mock_inference_fixture.query.return_value = (
            "# Title\nThis is **bold** and *italic*."
        )

        agent = TranslatorAgent(inference=mock_inference_fixture)
        agent.translate_input_to_english(original_text)

        mock_inference_fixture.query.assert_called_once()
        args, kwargs = mock_inference_fixture.query.call_args
        prompt = args[0] if args else kwargs.get("messages", "")

        assert "Translate the following text from fr to en." in prompt
        assert "crucial to preserve the original formatting exactly" in prompt
        assert "markdown syntax: headers" in prompt
        assert "bold (e.g., **text** or __text__)" in prompt
        assert "italics (e.g., *text* or _text_)" in prompt
        assert "links (e.g., [text](url))" in prompt
        assert "tables (using pipe and hyphen syntax)" in prompt
        # In this case, original_text has no code blocks, so it's passed as is to the prompt
        assert original_text in prompt

    @patch("fedotllm.agents.translator.detect")
    def test_unsupported_language_detection_still_calls_llm(
        self, mock_detect, mock_inference_fixture
    ):
        mock_detect.return_value = "xx"
        original_text = "Texte dans une langue inconnue."
        translated_by_llm = "Text in an unknown language, translated by LLM."
        mock_inference_fixture.query.return_value = translated_by_llm

        agent = TranslatorAgent(inference=mock_inference_fixture)
        translated_text = agent.translate_input_to_english(original_text)

        assert translated_text == translated_by_llm
        mock_inference_fixture.query.assert_called_once()
        args, kwargs = mock_inference_fixture.query.call_args
        prompt = args[0] if args else kwargs.get("messages", "")
        assert "Translate the following text from xx to en." in prompt
        assert original_text in prompt
