import json
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import Dataset
import torch

# Load the Q&A data
with open('policy_qa.json', 'r') as f:
    qa_data = json.load(f)

# Prepare the dataset
def prepare_data(qa_data):
    data = []
    for qa in qa_data:
        prompt = f"Question: {qa['question']}\nAnswer:"
        completion = qa['answer']
        data.append({"prompt": prompt, "completion": completion})
    return data

data = prepare_data(qa_data)
dataset = Dataset.from_list(data)

# Load a pre-trained model and tokenizer
model_name = "gpt2"  # Smaller model for fine-tuning
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Add padding token if not present
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Tokenize the dataset
def tokenize_function(examples):
    inputs = tokenizer(examples["prompt"], truncation=True, padding="max_length", max_length=512)
    outputs = tokenizer(examples["completion"], truncation=True, padding="max_length", max_length=512)
    inputs["labels"] = outputs["input_ids"]
    return inputs

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir="./logs",
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained("fine_tuned_model")
tokenizer.save_pretrained("fine_tuned_model")

print("Model trained and saved!")
