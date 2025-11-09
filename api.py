from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util
# Import Q&A data from separate files
try:
    from progressive_variable_pay_qa import progressive_variable_pay_qa
    from flexnxt_2_0_qa import flexnxt_2_0_qa
    from increment_policy_qa import increment_policy_qa
except ImportError:
    # Fallback if files not found
    progressive_variable_pay_qa = []
    flexnxt_2_0_qa = []
    increment_policy_qa = []
from pii_redactor import PIIRedactor

app = Flask(__name__)

# Load Q&A data from JSON file
import json
with open('policy_qa.json', 'r') as f:
    all_qa = json.load(f)

# Load local model for similarity
model = SentenceTransformer('all-MiniLM-L6-v2')

# Precompute embeddings for questions
questions = [qa['question'] for qa in all_qa]
question_embeddings = model.encode(questions, convert_to_tensor=True)

# Initialize PII redactor
redactor = PIIRedactor()

# Function to find the best matching answer using local similarity
def find_best_match(user_question, qa_list, question_embeddings):
    # Redact PII from user input
    redacted_question, _ = redactor.redact_input(user_question)

    # Encode the redacted question
    question_embedding = model.encode(redacted_question, convert_to_tensor=True)

    # Compute cosine similarities
    similarities = util.pytorch_cos_sim(question_embedding, question_embeddings)[0]

    # Find the best match
    best_idx = similarities.argmax().item()
    best_similarity = similarities[best_idx].item()

    # Threshold for relevance
    if best_similarity > 0.5:  # Adjust threshold as needed
        return {
            "answer": qa_list[best_idx]['answer'],
            "citations": [],  # No citations for Q&A matching
            "confidence": round(best_similarity, 2),
            "cached": False
        }
    else:
        return {
            "answer": "I couldn’t find a clear policy on this. Please check with HR or submit a ticket.",
            "citations": [],
            "confidence": 0.0,
            "cached": False
        }

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    if not data or 'question' not in data:
        return jsonify({"error": "Missing 'question' field"}), 400

    question = data['question']

    # No ambiguity check for simplicity

    result = find_best_match(question, all_qa, question_embeddings)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
