import time
import json
import os

class Metrics:
    def __init__(self, storage_file="metrics.json"):
        self.storage_file = storage_file
        self.latencies = []
        self.cache_hits = 0
        self.total_queries = 0
        self.model_costs = {}  # model: total_cost
        self.clarifications = 0
        self.pii_redactions = 0
        self.top_questions = {}  # question: count
        self.load()

    def load(self):
        if os.path.exists(self.storage_file):
            with open(self.storage_file, 'r') as f:
                data = json.load(f)
                self.latencies = data.get('latencies', [])
                self.cache_hits = data.get('cache_hits', 0)
                self.total_queries = data.get('total_queries', 0)
                self.model_costs = data.get('model_costs', {})
                self.clarifications = data.get('clarifications', 0)
                self.pii_redactions = data.get('pii_redactions', 0)
                self.top_questions = data.get('top_questions', {})

    def save(self):
        data = {
            'latencies': self.latencies,
            'cache_hits': self.cache_hits,
            'total_queries': self.total_queries,
            'model_costs': self.model_costs,
            'clarifications': self.clarifications,
            'pii_redactions': self.pii_redactions,
            'top_questions': self.top_questions
        }
        with open(self.storage_file, 'w') as f:
            json.dump(data, f)

    def log_latency(self, latency):
        self.latencies.append(latency)
        self.save()

    def log_cache_hit(self):
        self.cache_hits += 1
        self.save()

    def log_query(self):
        self.total_queries += 1
        self.save()

    def log_model_cost(self, model, cost):
        if model not in self.model_costs:
            self.model_costs[model] = 0
        self.model_costs[model] += cost
        self.save()

    def log_clarification(self):
        self.clarifications += 1
        self.save()

    def log_pii_redaction(self, count):
        self.pii_redactions += count
        self.save()

    def log_question(self, question):
        if question not in self.top_questions:
            self.top_questions[question] = 0
        self.top_questions[question] += 1
        self.save()

    def get_p95_latency(self):
        if not self.latencies:
            return 0
        sorted_latencies = sorted(self.latencies)
        index = int(0.95 * len(sorted_latencies))
        return sorted_latencies[index] if index < len(sorted_latencies) else sorted_latencies[-1]

    def get_cache_hit_rate(self):
        if self.total_queries == 0:
            return 0
        return self.cache_hits / self.total_queries

    def get_clarification_rate(self):
        if self.total_queries == 0:
            return 0
        return self.clarifications / self.total_queries

    def get_top_questions(self, n=10):
        return sorted(self.top_questions.items(), key=lambda x: x[1], reverse=True)[:n]

    def reset(self):
        self.latencies = []
        self.cache_hits = 0
        self.total_queries = 0
        self.model_costs = {}
        self.clarifications = 0
        self.pii_redactions = 0
        self.top_questions = {}
        self.save()
