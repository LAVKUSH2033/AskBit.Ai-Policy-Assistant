import openai

# Set your OpenAI API key here
openai.api_key = 'YOUR_API_KEY'  # Replace with your actual API key

def query_fine_tuned_model(question, model_name):
    """Query the fine-tuned model with a user question."""
    try:
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are a Q&A assistant for company policies. Answer only based on the provided Q&A pairs. If the question is not covered in the Q&A, respond with 'Sorry, I don't know the answer to that question.'"
                },
                {
                    "role": "user",
                    "content": question
                }
            ],
            max_tokens=500,
            temperature=0  # Low temperature for consistent answers
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"Error querying model: {e}"

if __name__ == "__main__":
    # Replace with your fine-tuned model name
    model_name = 'ft:gpt-3.5-turbo-0125:your-org:your-model-name'  # Update this

    while True:
        question = input("Ask a question (or 'quit' to exit): ")
        if question.lower() == 'quit':
            break
        answer = query_fine_tuned_model(question, model_name)
        print(f"Answer: {answer}")
