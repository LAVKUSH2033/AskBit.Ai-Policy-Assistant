# Project Title : AskBit.AI - Policy Assistant

### Team Name: Team 001(BitBuster)

---

## 🧩 Problem Statement Description

Build AskBit.AI, an intelligent internal policy copilot that answers employee questions about company policies, processes, and FAQs — accurately, securely, and fast — using only your organization’s official documents.
  

This project aims to create an **AI-powered Question & Answer system** that can instantly respond to employee policy-related queries in a conversational manner.

---

## 💡 Solution Overview

### Approach  
We built a **fine-tuned OpenAI Language Model (LLM)** using the organization’s policy Q&A data. The model is trained to provide accurate, context-aware answers to frequently asked questions about internal HR and pay policies.

### Architecture  
1. **Data Preparation:**  
   - Policy documents were structured into question-answer pairs (`policy_qa.json`).  
   - Data was converted to OpenAI fine-tuning format (`fine_tune_data.jsonl`).  

2. **Model Fine-Tuning:**  
   - Used the `gpt-3.5-turbo` base model for fine-tuning.  
   - Uploaded and trained using OpenAI’s fine-tuning API.

3. **Query Interface:**  
   - Developed a **Streamlit UI** for easy interaction.  
   - Also supports **CLI-based querying** for terminal users.

## Project Structure

- `policy_qa.json`: Original Q&A data.
- `convert_to_jsonl.py`: Script to convert JSON to JSONL format for fine-tuning.
- `fine_tune_data.jsonl`: Generated JSONL file for fine-tuning.
- `fine_tune_model.py`: Script to upload file and fine-tune the model.
- `query_model.py`: Script to query the fine-tuned model via CLI.
- `app.py`: Streamlit app for querying the model.
- `requirements.txt`: Dependencies.

### GenAI Model Used  
- **Model:** OpenAI GPT-3.5 Turbo , GitHub Copilot , ChatGPT , BlackBox.Ai
  
- **Purpose:** Domain-specific question answering for company policies.

---

## ⚙️ Environment Setup

### 1. Clone the Repository
```bash
git clone https://github.com/LAVKUSH2033/AskBit.Ai-Policy-Assistant.git
cd AskBit.Ai - Policy-Assistant

### 2.Install Dependencies
pip install -r requirements.txt

### 3.Configure API Key
Open the files fine_tune_model.py, query_model.py, and app.py
Replace YOUR_API_KEY
with your actual OpenAI API key


## 🚀 Execution Steps

### 1.Prepare the Dataset
Edit or add your company policy Q&A pairs in:
policy_qa.json

### 2. Convert Data to JSONL
python convert_to_jsonl.py

### 3.Fine-Tune the Model
python fine_tune_model.py

### 4.Test in CLI
python query_model.py

### 5.Launch Streamlit App
streamlit run app.py
or
Python -m streamlit run app.py

Access it at: http://localhost:8501

## ⚠️ Limitations and Future Enhancements
Currently, the model is trained on a limited set of policies. Future enhancements include:
- Expanding the dataset to cover more policies and edge cases.
- Implementing a feedback loop to improve answers over time.
- Adding support for multi-language queries.
- Enhancing the UI with features like query history and policy document previews.- Integrating with internal communication tools like Slack or Microsoft Teams for real-time assistance.- Enhancing the UI with features like query history and policy document previews.
- Integrating with internal communication tools like Slack or Microsoft Teams for real-time assistance.

##🏁 Conclusion:
This fine-tuned LLM solution provides an intelligent, accurate, and interactive way for employees to query company policies instantly.
It enhances transparency, accessibility, and efficiency in corporate communication.

##👤 Contributors
1. Lav Kush (9601)
2. Himanshu Prasad (9602)