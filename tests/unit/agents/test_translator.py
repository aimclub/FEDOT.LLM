import pytest
from unittest.mock import patch, MagicMock, ANY

# Assuming AIInference is in fedotllm.llm
from fedotllm.llm import AIInference 
from fedotllm.agents.translator import TranslatorAgent
from langdetect import LangDetectException

# No longer need MOCK_LANGUAGES from googletrans

@pytest.fixture
def mock_inference_fixture(): # Renamed to avoid conflict with parameter names in tests
    return MagicMock(spec=AIInference)

class TestTranslatorAgent:

    @patch('fedotllm.agents.translator.detect')
    def test_language_detection_success(self, mock_detect, mock_inference_fixture):
        mock_detect.return_value = 'es'
        agent = TranslatorAgent(inference=mock_inference_fixture)
        mock_inference_fixture.create.return_value = "Translated text" 
        agent.translate_input_to_english("Hola mundo")
        assert agent.source_language == 'es'
        mock_detect.assert_called_once_with("Hola mundo")

    @patch('fedotllm.agents.translator.detect')
    def test_language_detection_failure_defaults_to_english(self, mock_detect, mock_inference_fixture):
        mock_detect.side_effect = LangDetectException(0, "Detection failed")
        agent = TranslatorAgent(inference=mock_inference_fixture)
        mock_inference_fixture.create.return_value = "Translated text"
        agent.translate_input_to_english("Invalid text for detection")
        assert agent.source_language == 'en'
        mock_detect.assert_called_once_with("Invalid text for detection")

    @patch('fedotllm.agents.translator.detect')
    def test_translation_to_english_success(self, mock_detect, mock_inference_fixture):
        mock_detect.return_value = 'es'
        mock_inference_fixture.query.return_value = "Hello world"

        agent = TranslatorAgent(inference=mock_inference_fixture)
        translated_text = agent.translate_input_to_english("Hola mundo")
        
        assert translated_text == "Hello world"
        assert agent.source_language == 'es'
        mock_inference_fixture.query.assert_called_once()
        
        args, kwargs = mock_inference_fixture.query.call_args
        prompt = args[0] if args else kwargs.get('messages', '')
        assert "Translate the following text from es to en." in prompt
        assert "Hola mundo" in prompt

    @patch('fedotllm.agents.translator.detect')
    def test_translation_to_english_inference_error_returns_original(self, mock_detect, mock_inference_fixture):
        mock_detect.return_value = 'es'
        mock_inference_fixture.query.side_effect = Exception("LLM API error")

        agent = TranslatorAgent(inference=mock_inference_fixture)
        original_text = "Hola mundo"
        translated_text = agent.translate_input_to_english(original_text)
        
        assert translated_text == original_text 
        assert agent.source_language == 'es'

    @patch('fedotllm.agents.translator.detect')
    def test_english_input_not_translated(self, mock_detect, mock_inference_fixture):
        mock_detect.return_value = 'en'
        
        agent = TranslatorAgent(inference=mock_inference_fixture)
        input_text = "Hello world, this is English."
        translated_text = agent.translate_input_to_english(input_text)
        
        assert translated_text == input_text
        assert agent.source_language == 'en'
        mock_inference_fixture.query.assert_not_called()

    @patch('fedotllm.agents.translator.detect')
    def test_translation_to_source_language_success(self, mock_detect, mock_inference_fixture):
        mock_detect.return_value = 'es'
        
        mock_inference_fixture.query.return_value = "Hello world" 
        agent = TranslatorAgent(inference=mock_inference_fixture)
        agent.translate_input_to_english("Hola mundo")
        assert agent.source_language == 'es'
        
        mock_inference_fixture.query.reset_mock() 
        mock_inference_fixture.query.return_value = "Hola mundo otra vez"
        
        translated_output = agent.translate_output_to_source_language("Hello world again")
        
        assert translated_output == "Hola mundo otra vez"
        mock_inference_fixture.query.assert_called_once()
        args, kwargs = mock_inference_fixture.query.call_args
        prompt = args[0] if args else kwargs.get('messages', '')
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

    def test_reinsert_code_blocks(self, mock_inference_fixture):
        agent = TranslatorAgent(inference=mock_inference_fixture)
        text_with_placeholders = f"""Texto traducido
{agent.code_block_placeholder_prefix}_0__
Más texto traducido
{agent.code_block_placeholder_prefix}_1__"""
        code_map = {
            f"{agent.code_block_placeholder_prefix}_0__": "```python\nprint('hello')\n```",
            f"{agent.code_block_placeholder_prefix}_1__": "```\nx = 1\n```"
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


    @patch('fedotllm.agents.translator.detect')
    def test_code_block_preservation_e2e(self, mock_detect, mock_inference_fixture):
        mock_detect.return_value = 'es'
        original_text = """Texto antes
```python
# Esto es un comentario
print('Hola')
```
Texto después"""
        
        # Use the agent's placeholder prefix to construct the expected placeholder
        agent_for_placeholder = TranslatorAgent(inference=mock_inference_fixture) # Temp agent to get prefix if needed
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
        prompt = args[0] if args else kwargs.get('messages', '')
        assert f"placeholders like '{agent.code_block_placeholder_prefix}_NUMBER__'" in prompt
        assert "MUST NOT be translated or altered" in prompt
        assert placeholder_0 in prompt 

    @patch('fedotllm.agents.translator.detect')
    def test_markdown_preservation_prompting(self, mock_detect, mock_inference_fixture):
        mock_detect.return_value = 'fr'
        original_text = "# Titre\nCeci est du **gras** et de l'*italique*."
        
        mock_inference_fixture.query.return_value = "# Title\nThis is **bold** and *italic*."

        agent = TranslatorAgent(inference=mock_inference_fixture)
        agent.translate_input_to_english(original_text)

        mock_inference_fixture.query.assert_called_once()
        args, kwargs = mock_inference_fixture.query.call_args
        prompt = args[0] if args else kwargs.get('messages', '')

        assert "Translate the following text from fr to en." in prompt
        assert "crucial to preserve the original formatting exactly" in prompt
        assert "markdown syntax: headers" in prompt
        assert "bold (e.g., **text** or __text__)" in prompt
        assert "italics (e.g., *text* or _text_)" in prompt
        assert "links (e.g., [text](url))" in prompt
        assert "tables (using pipe and hyphen syntax)" in prompt
        # In this case, original_text has no code blocks, so it's passed as is to the prompt
        assert original_text in prompt 

    @patch('fedotllm.agents.translator.detect')
    def test_unsupported_language_detection_still_calls_llm(self, mock_detect, mock_inference_fixture):
        mock_detect.return_value = 'xx'
        original_text = "Texte dans une langue inconnue."
        translated_by_llm = "Text in an unknown language, translated by LLM."
        mock_inference_fixture.query.return_value = translated_by_llm

        agent = TranslatorAgent(inference=mock_inference_fixture)
        translated_text = agent.translate_input_to_english(original_text)

        assert translated_text == translated_by_llm
        mock_inference_fixture.query.assert_called_once()
        args, kwargs = mock_inference_fixture.query.call_args
        prompt = args[0] if args else kwargs.get('messages', '')
        assert "Translate the following text from xx to en." in prompt
        assert original_text in prompt
