import streamlit as st
from sentence_transformers import SentenceTransformer, util
import json

# Load Q&A data from JSON file
with open('policy_qa.json', 'r') as f:
    all_qa = json.load(f)

# Load local model for similarity
model = SentenceTransformer('all-MiniLM-L6-v2')

# Precompute embeddings for questions
questions = [qa['question'] for qa in all_qa]
question_embeddings = model.encode(questions, convert_to_tensor=True)

# Function to find the best matching answer using local similarity
def find_best_match(user_question, qa_list, question_embeddings):
    # Encode the question
    question_embedding = model.encode(user_question, convert_to_tensor=True)

    # Compute cosine similarities
    similarities = util.pytorch_cos_sim(question_embedding, question_embeddings)[0]

    # Find the best match
    best_idx = similarities.argmax().item()
    best_similarity = similarities[best_idx].item()

    # Threshold for relevance
    if best_similarity > 0.5:  # Adjust threshold as needed
        return qa_list[best_idx]['answer']
    else:
        return "I couldn’t find a clear policy on this. Please check with HR or submit a ticket."

st.title("ASKBIT.AI - Policy Assistant")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Ask a question about policies"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process normally
    with st.chat_message("assistant"):
        response = find_best_match(prompt, all_qa, question_embeddings)
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
