import time
import tiktoken
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
import openai
import os
import json
from metrics import Metrics

# Set up logging
logging.basicConfig(filename='failover.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ModelRouter:
    def __init__(self, openai_api_key=None):
        self.openai_api_key = openai_api_key or os.getenv("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")
        if self.openai_api_key:
            openai.api_key = self.openai_api_key

        # Load local fine-tuned model
        self.local_model_path = "fine_tuned_model"
        self.local_tokenizer = AutoTokenizer.from_pretrained(self.local_model_path)
        self.local_model = AutoModelForCausalLM.from_pretrained(self.local_model_path)

        # Model configurations with failover hierarchy
        self.models = {
            "local_gpt2": {
                "name": "Local GPT-2 Fine-tuned",
                "latency": 0.5,  # seconds
                "cost_per_token": 0.0,  # free
                "max_context": 1024,
                "type": "local",
                "failover_priority": 1  # Highest priority (lowest number)
            },
            "gpt-3.5-turbo": {
                "name": "GPT-3.5 Turbo",
                "latency": 1.0,
                "cost_per_token": 0.002 / 1000,  # $0.002 per 1K tokens
                "max_context": 4096,
                "type": "openai",
                "failover_priority": 2
            },
            "gpt-4": {
                "name": "GPT-4",
                "latency": 2.0,
                "cost_per_token": 0.03 / 1000,  # $0.03 per 1K tokens
                "max_context": 8192,
                "type": "openai",
                "failover_priority": 3
            }
        }

        # Cache for responses with versioning
        self.cache = {}
        self.document_version = "v1"  # Simple version tracking
        self.metrics = Metrics()
        self.metrics.log_query()  # Initialize total_queries

    def count_tokens(self, text, model="gpt-3.5-turbo"):
        try:
            encoding = tiktoken.encoding_for_model(model)
            return len(encoding.encode(text))
        except:
            # Fallback to approximate count
            return len(text.split()) * 1.3

    def select_model(self, question, context_length):
        """
        Route queries to the best LLM based on latency, cost, and context length.
        """
        question_tokens = self.count_tokens(question)
        total_tokens = question_tokens + context_length

        # Short questions (< 50 tokens) -> use cheapest/fastest
        if question_tokens < 50:
            return "local_gpt2"

        # Long context (> 2000 tokens) -> use larger model
        if total_tokens > 2000:
            if self.openai_api_key:
                return "gpt-4"
            else:
                return "local_gpt2"

        # Medium questions -> balance cost and performance
        if self.openai_api_key:
            return "gpt-3.5-turbo"
        else:
            return "local_gpt2"

    def generate_response_with_failover(self, primary_model, prompt, max_tokens=400, question=None):
        """
        Generate response with graceful degradation and failover.
        """
        # Create cache key with normalized question and document version
        normalized_question = question.lower().strip() if question else ""
        cache_key = f"{normalized_question}_{self.document_version}"
        if cache_key in self.cache:
            logging.info(f"Using warm cache for question: {normalized_question}")
            self.metrics.log_cache_hit()
            return self.cache[cache_key]

        # Try primary model first
        try:
            result = self.generate_response(primary_model, prompt, max_tokens)
            if "Error" not in result["response"]:
                self.cache[cache_key] = result
                self.metrics.log_latency(result["latency"])
                self.metrics.log_question(question)
                return result
        except Exception as e:
            logging.warning(f"Primary model {primary_model} failed: {str(e)}")

        # Failover to secondary models in priority order
        sorted_models = sorted(self.models.items(), key=lambda x: x[1]["failover_priority"])
        for model_name, model_config in sorted_models:
            if model_name != primary_model:
                try:
                    logging.info(f"Attempting failover to {model_name}")
                    result = self.generate_response(model_name, prompt, max_tokens)
                    if "Error" not in result["response"]:
                        logging.info(f"Failover successful with {model_name}")
                        self.cache[cache_key] = result
                        return result
                except Exception as e:
                    logging.warning(f"Failover model {model_name} also failed: {str(e)}")

        # Check cache again (in case of previous successful responses)
        if cache_key in self.cache:
            logging.info(f"Using cached response after failover attempts")
            return self.cache[cache_key]

        # Final fallback
        fallback_response = {
            "response": "Service temporarily slow — try again in 30s",
            "latency": 0.0,
            "cost": 0.0,
            "model": "fallback"
        }
        logging.error("All models failed, using fallback response")
        return fallback_response

    def clear_cache_on_document_update(self):
        """
        Clear cache when documents are updated.
        """
        self.document_version = f"v{int(self.document_version[1:]) + 1}"
        self.cache.clear()
        logging.info(f"Cache cleared due to document update. New version: {self.document_version}")

    def generate_response(self, model_name, prompt, max_tokens=400):
        start_time = time.time()

        if self.models[model_name]["type"] == "local":
            inputs = self.local_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.models[model_name]["max_context"])
            outputs = self.local_model.generate(
                inputs['input_ids'],
                max_length=max_tokens,
                num_return_sequences=1,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                top_k=40,
                no_repeat_ngram_size=2,
                pad_token_id=self.local_tokenizer.eos_token_id,
                eos_token_id=self.local_tokenizer.eos_token_id
            )
            response = self.local_tokenizer.decode(outputs[0], skip_special_tokens=True)
            latency = time.time() - start_time
            cost = 0.0

        elif self.models[model_name]["type"] == "openai":
            try:
                response_obj = openai.ChatCompletion.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=0.7,
                    timeout=10  # 10 second timeout
                )
                response = response_obj.choices[0].message.content
                latency = time.time() - start_time
                tokens_used = response_obj.usage.total_tokens
                cost = tokens_used * self.models[model_name]["cost_per_token"]
                self.metrics.log_model_cost(model_name, cost)
            except openai.error.RateLimitError:
                raise Exception("Rate limit exceeded")
            except openai.error.Timeout:
                raise Exception("Request timeout")
            except openai.error.APIError as e:
                if e.http_status == 500:
                    raise Exception("Internal server error")
                else:
                    raise Exception(f"API error: {str(e)}")
            except Exception as e:
                raise Exception(f"OpenAI error: {str(e)}")

        return {
            "response": response,
            "latency": round(latency, 2),
            "cost": round(cost, 4),
            "model": model_name
        }
