import re
import json
from metrics import Metrics

class ClarificationHandler:
    def __init__(self, qa_data_path='policy_qa.json'):
        with open(qa_data_path, 'r') as f:
            self.qa_data = json.load(f)
        self.metrics = Metrics()

        # Ambiguous question patterns
        self.ambiguous_patterns = {
            'pto': [
                r'\b(pto|paid time off|vacation time|leave)\b',
                r'\b(time off|vacation|leave)\b'
            ],
            'benefits': [
                r'\b(benefits|insurance|health|dental|vision)\b',
                r'\b(medical|healthcare|coverage)\b'
            ],
            'salary': [
                r'\b(salary|pay|compensation|wage)\b',
                r'\b(how much|what.*pay|compensation)\b'
            ],
            'location': [
                r'\b(location|office|remote|work from home)\b',
                r'\b(where.*work|office location)\b'
            ],
            'role': [
                r'\b(role|position|job|title)\b',
                r'\b(what.*do|responsibilities|duties)\b'
            ]
        }

        # Clarification questions
        self.clarification_questions = {
            'pto': "Are you asking about PTO for full-time employees, contractors, or a specific role/region?",
            'benefits': "Are you asking about health benefits, retirement benefits, or other employee benefits?",
            'salary': "Are you asking about salary ranges, pay structure, or compensation for a specific role?",
            'location': "Are you asking about office locations, remote work policies, or relocation assistance?",
            'role': "Are you asking about job responsibilities, required qualifications, or career progression?"
        }

    def detect_ambiguity(self, question):
        """
        Detect if a question is ambiguous and needs clarification.
        Returns the type of ambiguity if detected, None otherwise.
        """
        question_lower = question.lower()

        for ambiguity_type, patterns in self.ambiguous_patterns.items():
            for pattern in patterns:
                if re.search(pattern, question_lower):
                    # Check if question is too vague (short and contains ambiguity)
                    if len(question.split()) < 10:
                        self.metrics.log_clarification()
                        return ambiguity_type

        return None

    def get_clarification_question(self, ambiguity_type):
        """
        Get the appropriate clarification question for the ambiguity type.
        """
        return self.clarification_questions.get(ambiguity_type, "Could you please provide more details about your question?")

    def is_clarification_response(self, user_input, original_question):
        """
        Check if the user input is a response to a clarification question.
        """
        # Simple check - if input is short and contains keywords
        clarification_keywords = [
            'full-time', 'contractor', 'specific role', 'region', 'health', 'retirement',
            'salary range', 'pay structure', 'office', 'remote', 'responsibilities', 'qualifications'
        ]

        user_lower = user_input.lower()
        return any(keyword in user_lower for keyword in clarification_keywords) or len(user_input.split()) < 20

    def refine_question(self, original_question, clarification_response):
        """
        Refine the original question based on the clarification response.
        """
        # Simple concatenation for now - could be enhanced with NLP
        return f"{original_question} ({clarification_response})"
