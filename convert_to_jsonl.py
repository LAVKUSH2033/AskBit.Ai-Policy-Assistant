import json

# Load the Q&A data from policy_qa.json
with open('policy_qa.json', 'r', encoding='utf-8') as f:
    qa_list = json.load(f)

# Create the JSONL file for fine-tuning
with open('fine_tune_data.jsonl', 'w', encoding='utf-8') as f:
    for qa in qa_list:
        # Each training example is a conversation with system, user, assistant
        messages = [
            {
                "role": "system",
                "content": "You are a Q&A assistant for company policies. Answer only based on the provided Q&A pairs. If the question is not covered in the Q&A, respond with 'Sorry, I don't know the answer to that question.'"
            },
            {
                "role": "user",
                "content": qa['question']
            },
            {
                "role": "assistant",
                "content": qa['answer']
            }
        ]
        # Write each example as a JSON line
        f.write(json.dumps({"messages": messages}, ensure_ascii=False) + '\n')

    # Add a few examples for unknown questions to teach the fallback
    unknown_examples = [
        {
            "question": "What is the weather today?",
            "answer": "Sorry, I don't know the answer to that question."
        },
        {
            "question": "How to apply for leave?",
            "answer": "Sorry, I don't know the answer to that question."
        }
    ]
    for qa in unknown_examples:
        messages = [
            {
                "role": "system",
                "content": "You are a Q&A assistant for company policies. Answer only based on the provided Q&A pairs. If the question is not covered in the Q&A, respond with 'Sorry, I don't know the answer to that question.'"
            },
            {
                "role": "user",
                "content": qa['question']
            },
            {
                "role": "assistant",
                "content": qa['answer']
            }
        ]
        f.write(json.dumps({"messages": messages}, ensure_ascii=False) + '\n')

print("JSONL file 'fine_tune_data.jsonl' created successfully.")
