import yaml

class ContextManager:
    def __init__(self):
        with open('../config/settings.yaml') as f:
            self.config = yaml.safe_load(f)
        
        self.history = []
    
    def update_context(self, query, response):
        self.history.append((query, response))
        if len(self.history) > self.config['max_history']:
            self.history.pop(0)
    
    def get_context(self):
        return " ".join([f"Previous: {q} {r}" for q, r in self.history])

class MentalHealthContext(ContextManager):
    def emergency_check(self, query):
        emergency_triggers = ["help me die", "want to hurt myself"]
        return any(trigger in query.lower() for trigger in emergency_triggers)
    
    def get_context(self):
        base_context = super().get_context()
        return f"MENTAL HEALTH CONTEXT: {base_context} DISCLAIMER: I'm not a doctor."