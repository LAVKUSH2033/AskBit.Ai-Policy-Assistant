from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load the fine-tuned model
model_path = "fine_tuned_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

def query_model(question):
    prompt = f"Question: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs["input_ids"], max_length=200, num_return_sequences=1, temperature=0.7)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the answer part
    if "Answer:" in answer:
        answer = answer.split("Answer:")[1].strip()
    return answer

if __name__ == "__main__":
    while True:
        question = input("Ask a question (or 'quit' to exit): ")
        if question.lower() == 'quit':
            break
        answer = query_model(question)
        print(f"Answer: {answer}")
