import re
from typing import Optional

# Lazy-loaded Presidio analyzer and anonymizer
_analyzer = None
_anonymizer = None

def get_presidio_engines():
    """Initialize Presidio Analyzer and Anonymizer engines."""
    global _analyzer, _anonymizer
    if _analyzer is None or _anonymizer is None:
        try:
            from presidio_analyzer import AnalyzerEngine
            from presidio_anonymizer import AnonymizerEngine
            _analyzer = AnalyzerEngine()
            _anonymizer = AnonymizerEngine()
        except Exception as e:
            print(f"Warning: Presidio initialization fallback ({e})")
            _analyzer = False
            _anonymizer = False
    return _analyzer, _anonymizer

# Regex fallbacks for standard PII patterns
SSN_REGEX = re.compile(r'\b\d{3}-\d{2}-\d{4}\b')
PHONE_REGEX = re.compile(r'\b(?:\+?1[-. ]?)?\(?[2-9]\d{2}\)?[-. ]?\d{3}[-. ]?\d{4}\b')
EMAIL_REGEX = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b')

def fallback_regex_mask(text: str) -> str:
    """Fallback regex masking for standard sensitive patterns."""
    text = SSN_REGEX.sub("<SSN>", text)
    text = PHONE_REGEX.sub("<PHONE_NUMBER>", text)
    text = EMAIL_REGEX.sub("<EMAIL_ADDRESS>", text)
    return text

def mask_pii(text: str) -> str:
    """
    Detect and mask sensitive PII (PERSON, SSN, PHONE, EMAIL, MEDICAL_LICENSE)
    using Microsoft Presidio, backed by regex fallbacks.
    Preserves document structure and clinical details while protecting patient privacy.
    """
    if not text or not isinstance(text, str):
        return ""

    analyzer, anonymizer = get_presidio_engines()

    if analyzer and anonymizer:
        try:
            results = analyzer.analyze(
                text=text,
                entities=[
                    "PERSON",
                    "US_SSN",
                    "PHONE_NUMBER",
                    "EMAIL_ADDRESS",
                    "MEDICAL_LICENSE",
                    "US_PASSPORT"
                ],
                language="en"
            )
            anonymized = anonymizer.anonymize(
                text=text,
                analyzer_results=results
            )
            return fallback_regex_mask(anonymized.text)
        except Exception as e:
            # Fallback to regex if spacy or presidio encounters an exception on unusual strings
            return fallback_regex_mask(text)

    return fallback_regex_mask(text)
