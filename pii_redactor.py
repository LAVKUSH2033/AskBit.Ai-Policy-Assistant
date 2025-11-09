import re
import spacy
from metrics import Metrics

class PIIRedactor:
    def __init__(self):
        # Load lightweight NER model (spaCy small model)
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            # If model not available, use regex only
            self.nlp = None
        self.metrics = Metrics()

        # Regex patterns for PII detection
        self.patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b',
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'credit_card': r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b',
            'address': r'\b\d+\s+[A-Za-z0-9\s,.-]+\b',  # Simple address pattern
            'zip_code': r'\b\d{5}(?:-\d{4})?\b'
        }

    def detect_and_redact(self, text):
        """
        Detect and redact PII from text.
        Returns redacted text and list of detected PII types.
        """
        redacted_text = text
        detected_pii = []

        # Regex-based detection
        for pii_type, pattern in self.patterns.items():
            matches = re.findall(pattern, text)
            if matches:
                detected_pii.extend([pii_type] * len(matches))
                # Replace with placeholder
                placeholder = f"[{pii_type.upper()}]"
                redacted_text = re.sub(pattern, placeholder, redacted_text)

        # NER-based detection (if spaCy available)
        if self.nlp:
            doc = self.nlp(text)
            for ent in doc.ents:
                if ent.label_ in ['PERSON', 'ORG', 'GPE', 'LOC']:
                    # Only redact if not already redacted by regex
                    if f"[{ent.label_}]" not in redacted_text[ent.start_char:ent.end_char]:
                        placeholder = f"[{ent.label_}]"
                        redacted_text = redacted_text[:ent.start_char] + placeholder + redacted_text[ent.end_char:]
                        detected_pii.append(ent.label_.lower())

        return redacted_text, detected_pii

    def redact_input(self, user_input):
        """
        Redact PII from user input before processing.
        """
        return self.detect_and_redact(user_input)

    def redact_output(self, model_output):
        """
        Redact PII from model output before display/logging.
        """
        return self.detect_and_redact(model_output)

    def log_redacted(self, original_text, redacted_text, detected_pii, source="input"):
        """
        Log redacted information for compliance.
        """
        if detected_pii:
            self.metrics.log_pii_redaction(len(detected_pii))
            with open('pii_log.txt', 'a') as f:
                f.write(f"{source.upper()} - Detected PII: {detected_pii}\n")
                f.write(f"Original: {original_text}\n")
                f.write(f"Redacted: {redacted_text}\n\n")
