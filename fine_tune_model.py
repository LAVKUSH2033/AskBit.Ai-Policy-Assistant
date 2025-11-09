import openai
import time

# Set your OpenAI API key here
openai.api_key = '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'

def upload_file(file_path):
    """Upload the JSONL file to OpenAI."""
    with open(file_path, 'rb') as f:
        response = openai.files.create(
            file=f,
            purpose='fine-tune'
        )
    return response.id

def create_fine_tune_job(file_id, model='gpt-3.5-turbo'):
    """Create a fine-tune job."""
    response = openai.fine_tuning.jobs.create(
        training_file=file_id,
        model=model
    )
    return response.id

def check_fine_tune_status(job_id):
    """Check the status of the fine-tune job."""
    response = openai.fine_tuning.jobs.retrieve(job_id)
    return response.status, response.fine_tuned_model

def fine_tune_model():
    """Main function to fine-tune the model."""
    # Step 1: Upload the file
    print("Uploading file...")
    file_id = upload_file('fine_tune_data.jsonl')
    print(f"File uploaded with ID: {file_id}")

    # Step 2: Create fine-tune job
    print("Creating fine-tune job...")
    job_id = create_fine_tune_job(file_id)
    print(f"Fine-tune job created with ID: {job_id}")

    # Step 3: Monitor the job
    print("Monitoring fine-tune job...")
    while True:
        status, model_name = check_fine_tune_status(job_id)
        print(f"Status: {status}")
        if status == 'succeeded':
            print(f"Fine-tuning completed. Model name: {model_name}")
            return model_name
        elif status == 'failed':
            raise Exception("Fine-tuning failed.")
        time.sleep(60)  # Check every minute

if __name__ == "__main__":
    try:
        fine_tuned_model = fine_tune_model()
        print(f"Use this model name for querying: {fine_tuned_model}")
    except Exception as e:
        print(f"Error: {e}")
