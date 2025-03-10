import yaml
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import networkx as nx  # For knowledge graph
from sklearn.metrics.pairwise import cosine_similarity

class SemanticSearch:
    def __init__(self):
        with open('../config/settings.yaml') as f:
            self.config = yaml.safe_load(f)

        self.model = SentenceTransformer(self.config['model'])
        self.data = pd.read_csv(self.config['data_path']).fillna('')  
        self._precompute_embeddings()

    def _precompute_embeddings(self):
        if "question" not in self.data.columns:
            raise ValueError("Data file must contain a 'question' column.")

        self.embeddings = self.model.encode(
            self.data['question'].tolist(),
            convert_to_tensor=True
        ) 

    def search(self, query, context=[]):
        context_messages = context
        weighted_context = ' '.join([msg * int(1 + i / len(context_messages)) for i, msg in enumerate(context_messages)])
        full_query = f"{weighted_context} {query}".strip()
        query_embedding = self.model.encode(full_query, convert_to_tensor=True)
        cos_scores = util.cos_sim(query_embedding, self.embeddings)[0]
        
        return {
            'scores': cos_scores,
            'answers': self.data['answer'].tolist()  # Return ALL answers
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
                "response": "Hello! How can I assist you today?",
                "threshold": 0.85
            }
        }
        self.knowledge_graph = self._load_knowledge_graph()
        self.recommendations = self._load_recommendations()

    def _load_knowledge_graph(self):
        # Load or create a knowledge graph
        G = nx.Graph()
        # Example: Add nodes and edges
        G.add_edge("mental health", "therapy")
        G.add_edge("therapy", "cognitive behavioral therapy")
        return G 

    def _load_recommendations(self):
        recommendations = {
            "stress": ["meditation", "exercise", "talk to a friend"],
            "anxiety": ["deep breathing", "journaling", "therapy"]
        }
        return recommendations

    def _expand_query_with_knowledge_graph(self, query):
        expanded_query = query
        for word in query.split():
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
        
        return None, 2

    def _get_recommendations(self, query):
        for key, recs in self.recommendations.items():
            if key in query:
                return recs
        return []

    def search(self, query, context=[]):
        expanded_query = self._expand_query_with_knowledge_graph(query)
        intent, intent_score = self._match_special_intent(expanded_query)
        semantic_results = super().search(expanded_query, context)
        
        if intent and intent_score > torch.max(semantic_results['scores']).item():
            # Return the special intent response if it has a higher score
            return {   
                'scores': torch.tensor([intent_score]),
                'answers': [self.special_intents[intent]["response"]]
            }
        
        # Filter and return the top result from the semantic search
        top_answer_idx = torch.argmax(semantic_results['scores']).item()
        recommendations = self._get_recommendations(query)
        return {
            'scores': semantic_results['scores'][top_answer_idx:top_answer_idx + 1],
            'answers': [semantic_results['answers'][top_answer_idx]],
            'recommendations': recommendations
        }

# x = MentalHealthSearch()

# context_messages = []

# while True:
#     user_input = input("Enter your query (type 'q' to quit): ")
#     if user_input.lower() == 'q':
#         break

#     context_messages.append(user_input)
#     if len(context_messages) > 2:
#         context_messages.pop(0)

#     context = " ".join(context_messages)
#     response = x.search(user_input, context)
#     print(response)
