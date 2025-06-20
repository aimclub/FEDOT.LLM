import re

from langdetect import LangDetectException, detect

from fedotllm.llm import AIInference
from fedotllm.log import logger


class TranslatorAgent:
    def __init__(self, inference: AIInference):
        self.source_language = None
        self.inference = inference
        self.code_block_placeholder_prefix = "__CODE_BLOCK_PLACEHOLDER__"
        logger.info("TranslatorAgent initialized with provided AIInference instance.")

    def _extract_code_blocks(self, text: str) -> tuple[str, dict[str, str]]:
        code_blocks_map = {}
        pattern = re.compile(r"(```(?:[a-zA-Z0-9_.-]+)?\n)(.*?)(\n```)", re.DOTALL)
        idx = 0

        def replace_match(match):
            nonlocal idx
            placeholder = f"{self.code_block_placeholder_prefix}_{idx}__"
            code_blocks_map[placeholder] = match.group(0)
            idx += 1
            return placeholder

        processed_text = pattern.sub(replace_match, text)
        if idx > 0:
            logger.debug(
                f"_extract_code_blocks: Found and replaced {idx} code blocks. Placeholders: {list(code_blocks_map.keys())}"
            )
        else:
            logger.debug("_extract_code_blocks: No code blocks found in text.")
        return processed_text, code_blocks_map

    def _reinsert_code_blocks(self, text: str, code_blocks_map: dict[str, str]) -> str:
        if not code_blocks_map:
            logger.debug("_reinsert_code_blocks: No code blocks in map to reinsert.")
            return text

        logger.debug(
            f"_reinsert_code_blocks: Attempting to reinsert {len(code_blocks_map)} code block(s) from map: {list(code_blocks_map.keys())}"
        )
        reinserted_count = 0
        not_found_placeholders = []
        for placeholder, original_code_block in code_blocks_map.items():
            escaped_placeholder = re.escape(placeholder)
            text_after_sub, num_replacements = re.subn(
                escaped_placeholder, lambda m: original_code_block, text, count=1
            )
            logger.debug(
                f"_reinsert_code_blocks: Searching for placeholder: '{placeholder}'. Found and replaced: {num_replacements > 0} (Count: {num_replacements})"
            )
            if num_replacements > 0:
                reinserted_count += 1
                text = text_after_sub
            else:
                logger.warning(
                    f"_reinsert_code_blocks: Placeholder '{placeholder}' not found in the translated text for re-insertion."
                )
                not_found_placeholders.append(placeholder)
        if reinserted_count == len(code_blocks_map) and not not_found_placeholders:
            logger.debug(
                f"_reinsert_code_blocks: Successfully reinserted all {reinserted_count} code blocks."
            )
        else:
            logger.warning(
                f"_reinsert_code_blocks: Reinserted {reinserted_count} out of {len(code_blocks_map)} code blocks from map. "
                f"Placeholders not found in text: {not_found_placeholders if not_found_placeholders else 'None (but counts mismatch, check logic)'}"
            )
        return text

    def _translate_text(
        self, text: str, target_language: str, source_language: str = "auto"
    ) -> str:
        logger.info(
            f"TranslatorAgent._translate_text: Attempting translation from '{source_language}' to '{target_language}' using self.inference.query."
        )
        logger.debug(
            f"Original text for _translate_text (first 200 chars): '{text[:200]}...'"
        )

        # If the input text is empty, return it directly
        if not text:
            logger.debug(
                "Input text to _translate_text is empty. Returning empty string."
            )
            return ""

        processed_text, code_blocks_map = self._extract_code_blocks(text)
        if text != processed_text:
            logger.debug(
                f"Text after code block extraction (first 200 chars): '{processed_text[:200]}...'"
            )

        prompt_source_lang_description = source_language  # Default
        if str(source_language).lower() == "auto" or source_language is None:
            prompt_source_lang_description = "the auto-detected source language"
        elif str(source_language).lower() == "en":
            prompt_source_lang_description = "English"
        else:  # Use the language code directly for others
            prompt_source_lang_description = str(source_language)

        prompt = (
            f"Translate the following text from {prompt_source_lang_description} to {target_language}. "
            f"It is absolutely crucial to preserve the original formatting exactly. "
            f"This includes all markdown syntax: headers (e.g., #, ##), lists (e.g., -, *, 1.), "
            f"bold (e.g., **text** or __text__), italics (e.g., *text* or _text_), "
            f"strikethrough (e.g., ~~text~~), links (e.g., [text](url)), images (e.g., ![alt](url)), "
            f"tables (using pipe and hyphen syntax), and blockquotes (e.g., > text). "
            f"The text provided may contain placeholders like '{self.code_block_placeholder_prefix}_NUMBER__' "
            f"(e.g., {self.code_block_placeholder_prefix}_0__, {self.code_block_placeholder_prefix}_1__). "
            f"These placeholders represent original code blocks and MUST NOT be translated or altered in any way. "
            f"They must be preserved exactly as they appear in the input text. "
            f"Only translate the surrounding text. "
            f"If the text (excluding placeholders) is already in {target_language} and requires no translation, "
            f"return it as is, ensuring placeholders are also returned as is.\n\n"
            f"Text to translate (placeholders like {self.code_block_placeholder_prefix}_0__ must be kept as is):\n{processed_text}"
        )

        try:
            logger.debug(
                f"TranslatorAgent: Sending prompt to self.inference.query (from '{source_language}' to '{target_language}'):\n{prompt}"
            )
            response_text = self.inference.query(prompt)
            logger.debug(
                f"TranslatorAgent: Received response from self.inference.query. Type: {type(response_text)}. Content (first 200 chars): '{str(response_text)[:200]}...'"
            )

            if isinstance(response_text, str):
                if (
                    response_text == "" and processed_text != ""
                ):  # LLM returned empty string for non-empty input
                    logger.info(
                        "LLM returned an empty string. Assuming it's an intentional empty translation."
                    )
                    translated_text_with_placeholders = (
                        ""  # Use empty string for re-insertion
                    )
                elif response_text == processed_text:
                    logger.info(
                        f"Text from {source_language} to {target_language} may not have been translated by LLM (output is same as input to LLM)."
                    )
                    translated_text_with_placeholders = response_text
                else:  # Non-empty, different from processed_text
                    translated_text_with_placeholders = response_text
                    logger.info(
                        f"Successfully translated text from {source_language} to {target_language} using self.inference.query."
                    )
            else:  # Not a string (None, int, etc.)
                logger.warning(
                    f"LLM query did not return a valid string (type: {type(response_text)}, value: '{str(response_text)[:200]}...'). Returning original text."
                )
                return text  # Return original, unprocessed text

        except Exception as e:
            logger.error(
                f"Error during translation using self.inference.query from '{source_language}' to '{target_language}': {e}",
                exc_info=True,
            )
            return text  # Return original, unprocessed text

        logger.debug(
            f"TranslatorAgent._translate_text: Text before re-inserting code blocks (placeholders should be visible):\n{translated_text_with_placeholders}"
        )
        final_translated_text = self._reinsert_code_blocks(
            translated_text_with_placeholders, code_blocks_map
        )
        logger.debug(
            f"Final translated text after reinserting code blocks (target: {target_language}, first 200 chars): '{final_translated_text[:200]}...'"
        )
        return final_translated_text

    def translate_input_to_english(self, message: str) -> str:
        logger.info(
            f"TranslatorAgent: Received input message for translation to English (first 200 chars): '{message[:200]}...'"
        )
        if not message:
            logger.info("Input message is empty. Skipping detection and translation.")
            self.source_language = None  # Explicitly set, as no detection happens
            return ""
        try:
            self.source_language = detect(message)
        except LangDetectException as e:
            self.source_language = "en"  # Default if detection fails
            logger.warning(
                f"Language detection failed for input message (defaulting to 'en'): {e}",
                exc_info=True,
            )

        logger.info(
            f"TranslatorAgent: Source language for input set to: {self.source_language}"
        )

        if self.source_language != "en":
            logger.info(f"Translating input from {self.source_language} to English.")
            return self._translate_text(
                message, target_language="en", source_language=self.source_language
            )
        else:
            logger.info("Input is already English. No translation needed.")
            return message

    def translate_output_to_source_language(self, message: str) -> str:
        logger.info(
            f"TranslatorAgent: Attempting output translation. Current source_language: {self.source_language}"
        )

        if not message:  # Handles empty string, None, etc.
            logger.info("Output message is empty. Skipping translation.")
            return ""  # Return empty string consistently

        if self.source_language and self.source_language != "en":
            logger.info(f"Translating output from English to {self.source_language}.")
            logger.info(
                f"English message for output translation (first 200 chars): '{message[:200]}...'"
            )
            return self._translate_text(
                message, target_language=self.source_language, source_language="en"
            )
        elif not self.source_language:
            logger.warning(
                "Cannot translate output: source_language not set (input might not have been processed or detection failed)."
            )
            return message
        else:  # source_language is 'en'
            logger.info(
                f"Output translation not needed (source language was '{self.source_language}'). Returning original English message."
            )
            logger.debug(
                f"Original English message for output (first 200 chars): '{message[:200]}...'"
            )
            return message
