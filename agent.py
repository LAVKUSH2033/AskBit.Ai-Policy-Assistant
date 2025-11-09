import json
import difflib
from metrics import Metrics

class PolicyQAAgent:
    def __init__(self, qa_file='policy_qa.json'):
        with open(qa_file, 'r') as f:
            self.qa_data = json.load(f)
        self.questions = [qa['question'] for qa in self.qa_data]
        self.metrics = Metrics()

    def get_answer(self, user_question):
        self.metrics.log_question(user_question)
        # Find the closest match
        matches = difflib.get_close_matches(user_question, self.questions, n=1, cutoff=0.6)
        if matches:
            for qa in self.qa_data:
                if qa['question'] == matches[0]:
                    return qa['answer']
        return "Sorry, I don't know the answer to that question."

# Instantiate the agent
agent = PolicyQAAgent()
