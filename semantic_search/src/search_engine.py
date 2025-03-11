import yaml
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import networkx as nx  # For knowledge graph
from sklearn.metrics.pairwise import cosine_similarity

class SemanticSearch:
    def __init__(self):
        # Load configuration
        with open('../config/settings.yaml') as f:
            self.config = yaml.safe_load(f)

        self.model = SentenceTransformer(self.config['model'])  # all-MiniLM-L6-v2
        self.max_history = self.config['max_history']  # 3
        self.similarity_threshold = self.config['similarity_threshold']  # 0.4
        self.data = pd.read_csv(self.config['data_path']).fillna('')  #
        self._precompute_embeddings()

    def _precompute_embeddings(self):
        if "question" not in self.data.columns:
            raise ValueError("Data file must contain a 'question' column.")
        self.embeddings = self.model.encode(
            self.data['question'].tolist(),
            convert_to_tensor=True
        )

    def search(self, query, context=[]):
        # Combine context with weighted history
        if context:
            weighted_context = ' '.join([msg * int(1 + i / len(context)) for i, msg in enumerate(context)])
            full_query = f"{weighted_context} {query}".strip()
        else:
            full_query = query

        query_embedding = self.model.encode(full_query, convert_to_tensor=True)
        cos_scores = util.cos_sim(query_embedding, self.embeddings)[0]
        
        return {
            'scores': cos_scores,
            'answers': self.data['answer'].tolist()
        }

class MentalHealthSearch(SemanticSearch):
    def __init__(self):
        super().__init__()
        self.blocked_terms = {"suicide", "self-harm", "kill myself"}
        self.special_intents = {
            "bye": {
                "triggers": {"bye", "goodbye", "see ya", "later"},
                "response": "Take care and prioritize your mental health!",
                "threshold": 0.9
            },
            "greeting": {
                "triggers": {"hello", "hi", "hey"},
                "response": "Hello! How can I assist you with your mental health today?",
                "threshold": 0.85
            }
        }
        self.knowledge_graph = self._load_knowledge_graph()
        self.recommendations = self._load_recommendations()
        self.context_history = []  # To store session-based context

    def _load_knowledge_graph(self):
        # Build a simple mental health knowledge graph
        G = nx.Graph()
        G.add_edge("mental health", "therapy")
        G.add_edge("therapy", "cognitive behavioral therapy")
        G.add_edge("mental health", "stress")
        G.add_edge("stress", "anxiety")
        G.add_edge("anxiety", "panic attack")
        G.add_edge("mental health", "depression") 
        return G

    def _load_recommendations(self):
        # Mental health-specific recommendations
        return {
            "stress": ["Try meditation", "Take a short walk", "Talk to a friend"],
            "anxiety": ["Practice deep breathing", "Write in a journal", "Consider therapy"],
            "depression": ["Reach out to a loved one", "Engage in a hobby", "Seek professional help"]
        }

    def _expand_query_with_knowledge_graph(self, query):
        expanded_query = query.lower()
        for word in expanded_query.split():
            if word in self.knowledge_graph:
                neighbors = list(self.knowledge_graph.neighbors(word))
                expanded_query += ' ' + ' '.join(neighbors)
        return expanded_query

    def _match_special_intent(self, query):
        query_lower = query.lower()
        for intent, config in self.special_intents.items():
            triggers_present = [trigger for trigger in config['triggers'] if trigger in query_lower]
            if triggers_present:
                query_embed = self.model.encode(query, convert_to_tensor=True)
                intent_embed = self.model.encode(' '.join(config['triggers']), convert_to_tensor=True)
                score = util.cos_sim(query_embed, intent_embed).item()
                if score >= config['threshold']:  
                    return intent, score
        return None, 0  # Default score of 0 if no intent matches

    def _get_recommendations(self, query):
        query_lower = query.lower()
        for key, recs in self.recommendations.items():
            if key in query_lower:
                return recs
        return []

    def _check_blocked_terms(self, query):
        query_lower = query.lower()
        for term in self.blocked_terms:
            if term in query_lower:
                return True
        return False

    def search(self, query, context=[]):
        # Update context history
        self.context_history.append(query)
        if len(self.context_history) > self.max_history:
            self.context_history.pop(0)

        # Check for blocked terms
        if self._check_blocked_terms(query):
            return {
                'scores': torch.tensor([1.0]),
                'answers': ["I'm here to help. If you're in crisis, please contact a professional or helpline immediately."],
                'recommendations': ["Call a crisis hotline", "Talk to someone you trust"]
            }

        # Expand query with knowledge graph
        expanded_query = self._expand_query_with_knowledge_graph(query)
        
        # Check for special intents
        intent, intent_score = self._match_special_intent(expanded_query)
        semantic_results = super().search(expanded_query, self.context_history)

        if intent and intent_score > torch.max(semantic_results['scores']).item():
            return {
                'scores': torch.tensor([intent_score]),
                'answers': [self.special_intents[intent]["response"]],
                'recommendations': []
            }

        # Filter results based on similarity threshold
        top_indices = torch.where(semantic_results['scores'] >= self.similarity_threshold)[0]
        if len(top_indices) == 0:
            return {
                'scores': torch.tensor([0.0]),
                'answers': ["I couldn’t find a good match. Could you rephrase your query?"],
                'recommendations': []
            }

        top_score = semantic_results['scores'][top_indices[0]]
        top_answer = semantic_results['answers'][top_indices[0]]
        recommendations = self._get_recommendations(query)

        return {
            'scores': torch.tensor([top_score]),
            'answers': [top_answer],
            'recommendations': recommendations
        }

# Example usage
if __name__ == "__main__":
    search_engine = MentalHealthSearch()
    context_messages = []

    while True:
        user_input = input("Enter your query (type 'q' to quit): ")
        if user_input.lower() == 'q':
            break

        response = search_engine.search(user_input, context_messages)
        print(f"Answer: {response['answers'][0]}")
        print(f"Score: {response['scores'].item():.3f}")
        if response['recommendations']:
            print("Recommendations:", ", ".join(response['recommendations']))
        print("-" * 50)

        context_messages.append(user_input)
        if len(context_messages) > search_engine.max_history:
            context_messages.pop(0)