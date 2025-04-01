import spacy
from groq import Groq
import os

nlp = spacy.load("en_core_web_md")

# Keyword lists from knowledge graph
SYMPTOMS = ["fever", "cough", "headache", "fatigue", "nausea", "dizziness", "anxiety", "stomach upset", "insomnia"]
DISEASES = ["flu", "migraine", "depression", "anxiety", "fatigue", "insomnia", "stomach upset", "dizziness"]
TREATMENTS = ["rest", "fluids", "antiviral drugs", "pain relievers", "hydration", "dark room", "therapy", "antidepressants", "medications", "sleep hygiene", "dietary changes", "balanced diet"]

class GrokChat:
    def __init__(self, api_key):
        self.client = Groq(api_key=api_key or os.environ.get("GROQ_API_KEY"))

    def create_completion(self, prompt, max_tokens=200):
        try:
            completion = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile",
                max_tokens=max_tokens
            )
            response = completion.choices[0].message.content
            selected_q = response.split('\n')[0].replace("Selected Question: ", "").strip()
            answer = response.split('\n')[1].replace("Humanized Answer: ", "").strip()
            return selected_q, answer
        except Exception as e:
            return None, f"Error: Couldn’t generate response ({str(e)})"

def detect_intent(query):
    doc = nlp(query.lower())
    print(f"Tokens: {[f'{t.text}/{t.pos_}/{t.dep_}' for t in doc]}")

    if any(w in query.lower() for w in ["hi", "hello", "hey", "greetings"]):
        return "greeting"
    if "what" in query.lower() and any(w in query.lower() for w in ["wrong", "problem", "issue", "if i have"]):
        return "diagnosis"
    if any(w in query.lower() for w in ["how", "treat", "help", "manage", "cure"]):
        for token in doc:
            if token.dep_ == "ROOT" and token.pos_ in ["VERB", "AUX"]:
                return "treatment"
    if "what" in query.lower() and any(w in query.lower() for w in ["symptom", "symptoms", "sign", "signs"]):
        return "symptom"
    for token in doc:
        if token.pos_ in ["NOUN", "ADJ"] and token.dep_ in ["dobj", "nsubj", "attr"]:
            return "symptom"
    return "faq"

def extract_keyword(query, intent):
    """Extract the most relevant keyword based on intent and predefined lists."""
    doc = nlp(query.lower())
    query_words = [token.text for token in doc]
    
    if intent == "symptom":
        for word in query_words:
            if word in SYMPTOMS:
                return word
            if word in DISEASES:
                return word  # For "what is the symptom of migraine"
        return query_words[-1] if query_words else query  # Fallback to last word
    
    elif intent == "treatment":
        for word in query_words:
            if word in DISEASES:
                return word
            if word in TREATMENTS:
                return word  # For "what is the treatment of antiviral drugs"
        return query_words[-1] if query_words else query
    
    elif intent == "diagnosis":
        for word in query_words:
            if word in SYMPTOMS:
                return word
        return query_words[-1] if query_words else query
    
    return query

def chatbot(query, rag, llm):
    intent = detect_intent(query)
    print(f"Intent: {intent}")
    keyword = extract_keyword(query, intent)
    print(f"Extracted Keyword: {keyword}")
    prompt = ""

    try:
        if intent == "greeting":
            faq_answer = rag.query_faq(query)
            if faq_answer:
                prompt = f"Query: '{query}'\nFAQ Answer: '{faq_answer}'\nHumanize this for a medical chatbot with a friendly tone."
            else:
                prompt = f"Query: '{query}'\nNo FAQ found. Respond with a friendly greeting and offer help."

        elif intent == "symptom":
            diseases = rag.query_symptom(keyword)
            if diseases:
                disease = diseases[0]
                treatments = rag.get_treatment(disease)
                side_effects = [rag.get_side_effects(t) for t in treatments] if treatments else []
                prompt = (
                    f"Query: '{query}'\n"
                    f"Symptom or Disease: {keyword}\n"
                    f"Related Disease: {disease}\n"
                    f"Treatments: {', '.join(treatments) if treatments else 'None found'}\n"
                    f"Side Effects: {', '.join([', '.join(se) for se in side_effects if se]) or 'None noted'}\n"
                    "Generate a concise, empathetic response listing symptoms if a disease is queried."
                )
            else:
                prompt = f"Query: '{query}'\nNo diseases found for '{keyword}'. Suggest asking for more details."

        elif intent == "treatment":
            treatments = rag.get_treatment(keyword)
            if treatments:
                side_effects = [rag.get_side_effects(t) for t in treatments]
                prompt = (
                    f"Query: '{query}'\n"
                    f"Disease: {keyword}\n"
                    f"Treatments: {', '.join(treatments)}\n"
                    f"Side Effects: {', '.join([', '.join(se) for se in side_effects if se]) or 'None noted'}\n"
                    "Provide a helpful treatment suggestion with a caring tone."
                )
            else:
                prompt = f"Query: '{query}'\nNo treatments found for '{keyword}'. Offer general advice."

        elif intent == "diagnosis":
            diseases = rag.query_symptom(keyword)
            if diseases:
                prompt = (
                    f"Query: '{query}'\n"
                    f"Symptom: {keyword}\n"
                    f"Possible Diseases: {', '.join(diseases)}\n"
                    "Respond empathetically, suggesting a doctor visit if serious."
                )
            else:
                prompt = f"Query: '{query}'\nNo matching diseases found for '{keyword}'. Ask for more symptoms."

        elif intent == "faq":
            faq_answer = rag.query_faq(query)
            if faq_answer:
                prompt = f"Query: '{query}'\nFAQ Answer: '{faq_answer}'\nHumanize this for a medical chatbot with a friendly tone."
            else:
                prompt = f"Query: '{query}'\nNo FAQ match. Offer general medical assistance."

        if not prompt:
            prompt = f"Query: '{query}'\nNo specific info found. Offer general assistance with a friendly tone."

        prompt += "\nFormat response as: 'Selected Question: <question>' on first line, 'Humanized Answer: <response>' on second."
        selected_q, answer = llm.create_completion(prompt)
        return f"Selected Question: {selected_q or query}\nHumanized Answer: {answer}"

    except Exception as e:
        return f"Selected Question: {query}\nHumanized Answer: Sorry, something went wrong! ({str(e)})"