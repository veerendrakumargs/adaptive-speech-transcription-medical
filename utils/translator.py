
"""
Translation utilities for Indian regional languages.
Uses Gemini when GEMINI_API_KEY is set; otherwise falls back to Google Translate.
"""

import os
import time
import random

from deep_translator import GoogleTranslator

try:
    from dotenv import load_dotenv
    load_dotenv()

except ImportError:
    pass

try:
    from google import genai
except ImportError:
    genai = None


class IndianLanguageTranslator:
    """Translator for Indian regional languages."""

    INDIAN_LANGUAGES = {
        "hi": "Hindi (हिंदी)",
        "ta": "Tamil (தமிழ்)",
        "te": "Telugu (తెలుగు)",
        "ml": "Malayalam (മലയാളം)",
        "kn": "Kannada (ಕನ್ನಡ)",
        "bn": "Bengali (বাংলা)",
        "mr": "Marathi (मराठी)",
        "gu": "Gujarati (ગુજરાતી)",
        "pa": "Punjabi (ਪੰਜਾਬੀ)",
        "or": "Odia (ଓଡ଼ିଆ)",
        "as": "Assamese (অসমীয়া)",
        "ur": "Urdu (اردو)",
        "es": "Spanish (Español)",
        "fr": "French (Français)",
        "de": "German (Deutsch)",
    }

    MAX_RETRIES = 3
    MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.client = None

        if self.api_key and genai is not None:
            try:
                self.client = genai.Client(api_key=self.api_key)
            except Exception as error:
                print(f"Gemini setup failed: {error}")

    def _with_retries(self, func):
        """Retry a request with exponential backoff."""
        for attempt in range(self.MAX_RETRIES):
            try:
                result = func()
                if result:
                    return result
            except Exception as error:
                print(f"Attempt {attempt + 1} failed: {error}")

                if attempt < self.MAX_RETRIES - 1:
                    time.sleep((2 ** attempt) * 2 + random.uniform(0, 1))

                    return None

    def _gemini_translate(self, text, target_name, preserve_medical_terms):
        """Translate text using Gemini."""
        rule = ""

        if preserve_medical_terms:
            rule = (
                "Keep medical terms, drug names, dosages, units and abbreviations "
                "(for example MRI, ECG, HbA1c, metformin) in their original English form."
            )

            prompt = (
                f"Translate the following text into {target_name}. {rule} "
                "Keep numbers and formatting unchanged. "
                "Return only the translation, with no explanation.\n\n"
                f"{text}"
            )

            response = self.client.models.generate_content(
                model=self.MODEL,
                contents=prompt,
            )

            return (response.text or "").strip()

    def _google_translate(self, text, target_language):
        """Translate text using Google Translate in chunks."""
        translator = GoogleTranslator(
            source="auto",
            target=target_language,
        )

        parts = []

        for i in range(0, len(text), 4000):
            chunk = text[i:i + 4000]

            part = self._with_retries(
                lambda c=chunk: translator.translate(c)
            )

            if part is None:
                return None

            parts.append(part)

            if i + 4000 < len(text):
                time.sleep(1)

                return " ".join(parts)
    def translate_text(
        self,
        text,
        target_language,
        source_language="en",
        preserve_medical_terms=True,
):
        
        if not isinstance(text, str) or not text.strip():
            return text

        if target_language not in self.INDIAN_LANGUAGES:
            return f"[Error: Unsupported target language: {target_language}]"

        result = None
        target_name = self.INDIAN_LANGUAGES[target_language].split(" (")[0]

        if self.client:
            try:
                result = self._with_retries(
                    lambda: self._gemini_translate(
                        text, target_name, preserve_medical_terms
                    )
                )
            except Exception as error:
                print(f"Gemini translation failed: {error}")

                if result is None:
                    try:
                        result = self._google_translate(text, target_language)
                    except Exception as error:
                        print(f"Google translation failed: {error}")

                        if result is None:
                            return "[Translation failed. Please try again later.]"

                        return result

    def get_supported_languages(self):
        return self.INDIAN_LANGUAGES