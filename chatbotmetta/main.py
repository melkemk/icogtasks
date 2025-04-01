import os
from groq import Groq
from hyperon import MeTTa, E, S, ValueAtom
import spacy

nlp = spacy.load("en_core_web_md")
metta = MeTTa()

def initialize_knowledge_graph(metta):
    metta.space().add_atom(E(S("symptom"), S("fever"), S("flu")))
    metta.space().add_atom(E(S("symptom"), S("cough"), S("flu")))
    metta.space().add_atom(E(S("symptom"), S("headache"), S("migraine")))
    metta.space().add_atom(E(S("symptom"), S("fatigue"), S("depression")))
    metta.space().add_atom(E(S("symptom"), S("nausea"), S("flu")))
    metta.space().add_atom(E(S("symptom"), S("dizziness"), S("migraine")))
    metta.space().add_atom(E(S("symptom"), S("anxiety"), S("depression")))
    metta.space().add_atom(E(S("symptom"), S("stomach upset"), S("flu")))
    metta.space().add_atom(E(S("symptom"), S("insomnia"), S("depression"))) 
    
    # Diseases → Treatments 
    metta.space().add_atom(E(S("treatment"), S("flu"), ValueAtom("rest, fluids, antiviral drugs")))
    metta.space().add_atom(E(S("treatment"), S("migraine"), ValueAtom("pain relievers, hydration, dark room")))
    metta.space().add_atom(E(S("treatment"), S("depression"), ValueAtom("therapy, antidepressants")))
    metta.space().add_atom(E(S("treatment"), S("anxiety"), ValueAtom("therapy, medications")))  
    metta.space().add_atom(E(S("treatment"), S("fatigue"), ValueAtom("rest, hydration, balanced diet")))
    metta.space().add_atom(E(S("treatment"), S("insomnia"), ValueAtom("sleep hygiene, medications")))
    metta.space().add_atom(E(S("treatment"), S("stomach upset"), ValueAtom("dietary changes, medications")))
    metta.space().add_atom(E(S("treatment"), S("dizziness"), ValueAtom("hydration, medications")))
    metta.space().add_atom(E(S("treatment"), S("pain relievers"), ValueAtom("rest, hydration")))
    metta.space().add_atom(E(S("treatment"), S("antiviral drugs"), ValueAtom("rest, hydration")))
    metta.space().add_atom(E(S("treatment"), S("antidepressants"), ValueAtom("therapy, lifestyle changes")))
    metta.space().add_atom(E(S("treatment"), S("antidepressants"), ValueAtom("therapy, medications")))  
    
    # Treatments → Side Effects (hierarchy example)
    metta.space().add_atom(E(S("side_effect"), S("antiviral drugs"), ValueAtom("nausea, dizziness")))
    metta.space().add_atom(E(S("side_effect"), S("pain relievers"), ValueAtom("stomach upset")))
    metta.space().add_atom(E(S("side_effect"), S("antidepressants"), ValueAtom("weight gain, insomnia")))
    metta.space().add_atom(E(S("side_effect"), S("therapy"), ValueAtom("initial discomfort, emotional release")))
    metta.space().add_atom(E(S("side_effect"), S("lifestyle changes"), ValueAtom("initial discomfort, emotional release")))
    metta.space().add_atom(E(S("side_effect"), S("hydration"), ValueAtom("frequent urination, electrolyte imbalance")))
    metta.space().add_atom(E(S("side_effect"), S("dietary changes"), ValueAtom("bloating, gas")))
    metta.space().add_atom(E(S("side_effect"), S("sleep hygiene"), ValueAtom("initial discomfort, emotional release")))
    metta.space().add_atom(E(S("side_effect"), S("rest"), ValueAtom("initial discomfort, emotional release")))
    metta.space().add_atom(E(S("side_effect"), S("balanced diet"), ValueAtom("initial discomfort, emotional release")))
    metta.space().add_atom(E(S("side_effect"), S("dark room"), ValueAtom("initial discomfort, emotional release")))
    metta.space().add_atom(E(S("side_effect"), S("hydration"), ValueAtom("initial discomfort, emotional release")))
    metta.space().add_atom(E(S("side_effect"), S("pain relievers"), ValueAtom("initial discomfort, emotional release")))
    metta.space().add_atom(E(S("side_effect"), S("therapy"), ValueAtom("initial discomfort, emotional release")))
    metta.space().add_atom(E(S("side_effect"), S("medications"), ValueAtom("initial discomfort, emotional release")))
    metta.space().add_atom(E(S("side_effect"), S("therapy"), ValueAtom("initial discomfort, emotional release")))   
    
    # FAQs (from mental_health_faqs.csv-like structure)
    metta.space().add_atom(E(S("faq"), S("Hi"), ValueAtom("Hello! How can I assist you today?")))
    metta.space().add_atom(E(S("faq"), S("What’s wrong with me?"), ValueAtom("I’m not a doctor, but I can help you explore symptoms. What are you feeling?")))
    metta.space().add_atom(E(S("faq"), S("How do I treat a migraine?"), ValueAtom("For migraines, rest in a dark room and stay hydrated. Pain relievers can help.")))
    metta.space().add_atom(E(S("faq"), S("How do I treat flu?"), ValueAtom("For flu, rest and drink fluids. Antiviral drugs may help.")))
    metta.space().add_atom(E(S("faq"), S("How do I treat depression?"), ValueAtom("For depression, therapy and antidepressants can be effective.")))
    metta.space().add_atom(E(S("faq"), S("How do I treat anxiety?"), ValueAtom("For anxiety, therapy and medications can help.")))
    metta.space().add_atom(E(S("faq"), S("How do I treat fatigue?"), ValueAtom("For fatigue, rest and hydration are important.")))
    metta.space().add_atom(E(S("faq"), S("How do I treat insomnia?"), ValueAtom("For insomnia, sleep hygiene and medications can help.")))
    metta.space().add_atom(E(S("faq"), S("How do I treat stomach upset?"), ValueAtom("For stomach upset, dietary changes and medications can help.")))
    metta.space().add_atom(E(S("faq"), S("How do I treat dizziness?"), ValueAtom("For dizziness, hydration and medications can help.")))
    metta.space().add_atom(E(S("faq"), S("How do I treat pain relievers?"), ValueAtom("For pain relievers, rest and hydration are important.")))

class MedicalRAG:
    def __init__(self, metta_instance):
        self.metta = metta_instance

    def query_symptom(self, symptom):
        """Find diseases linked to a symptom."""
        symptom = symptom.strip('"')
        query_str = f'!(match &self (symptom {symptom} $disease) $disease)'
        print(symptom, 'symptom')
        print(query_str, '222')  
        results = self.metta.run(query_str)
        print(results, 222)
        unique_diseases = list(set(str(r[0]) for r in results if r and len(r) > 0)) if results else []
        return unique_diseases

    def get_treatment(self, disease):
        """Find treatments for a disease."""
        disease = disease.strip('"')
        query_str = f'!(match &self (treatment {disease} $treatment) $treatment)'
        print(disease, 'disease')
        print(query_str, '111')
        results = self.metta.run(query_str)
        print(results, 111)
        return [r[0].get_object().value for r in results if r and len(r) > 0] if results else []

    def get_side_effects(self, treatment):
        """Find side effects of a treatment."""
        treatment = treatment.strip('"')
        query_str = f'!(match &self (side_effect {treatment} $effect) $effect)'
        results = self.metta.run(query_str)
        return [r[0].get_object().value for r in results if r and len(r) > 0] if results else []

    def query_faq(self, question):
        """Retrieve FAQ answers."""
        query_str = f'!(match &self (faq "{question}" $answer) $answer)'
        results = self.metta.run(query_str)
        return results[0][0].get_object().value if results and results[0] else None

    def add_knowledge(self, relation_type, subject, object_value):
        """Add new knowledge dynamically."""
        if isinstance(object_value, str):
            object_value = ValueAtom(object_value)
        self.metta.space().add_atom(E(S(relation_type), S(subject), object_value))
        return f"Added {relation_type}: {subject} → {object_value}"

class GrokChat:
    def __init__(self, api_key):
        self.client = Groq(api_key=api_key or os.environ.get("GROQ_API_KEY"))

    def create_completion(self, prompt, max_tokens=200):
        completion = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            max_tokens=max_tokens
        )
        response = completion.choices[0].message.content
        try:
            selected_q = response.split('\n')[0].replace("Selected Question: ", "").strip()
            answer = response.split('\n')[1].replace("Humanized Answer: ", "").strip()
            return selected_q, answer
        except IndexError:
            return None, response

def detect_intent(query): 
    """Detect the intent of a query using spaCy."""
    doc = nlp(query.lower())
    print(f"Tokens: {[f'{t.text}/{t.pos_}/{t.dep_}' for t in doc]}")
    
    if any(w in query for w in ["hi", "hello", "hey"]):
        return "greeting"
    if "what" in query and any(w in query for w in ["wrong", "problem", "issue"]):
        return "diagnosis"
    if any(w in query for w in ["how", "treat", "help", "manage"]):
        for token in doc:
            if token.dep_ == "ROOT" and token.pos_ in ["VERB", "AUX"]:
                return "treatment"
    for token in doc:
        if token.pos_ in ["NOUN", "ADJ"] and token.dep_ in ["dobj", "nsubj", "attr"]:
            return "symptom"
    return "unknown" 

def chatbot(query, rag, llm):
    intent = detect_intent(query)
    print(f"Intent: {intent}")
    prompt = ""

    if intent == "greeting":
        faq_answer = rag.query_faq(query)
        if faq_answer:
            prompt = (
                f"Query: '{query}'\n"
                f"FAQ Answer: '{faq_answer}'\n"
                "Humanize this for a medical chatbot with a friendly tone."
            )  
    elif intent == "symptom":
        doc = nlp(query.lower())
        symptom = next((token.text for token in doc if token.pos_ in ["NOUN", "ADJ"] and token.dep_ in ["dobj", "nsubj", "attr"]), query)
        diseases = rag.query_symptom(symptom)  # No extra quotes
        if diseases:
            disease = diseases[0]
            treatments = rag.get_treatment(disease)  # No extra quotes
            side_effects = [rag.get_side_effects(t) for t in treatments] if treatments else []
            prompt = (
                f"Query: '{query}'\n"
                f"Symptom: {symptom}\n"
                f"Related Disease: {disease}\n"
                f"Treatments: {', '.join(treatments)}\n"
                f"Side Effects: {', '.join([', '.join(se) for se in side_effects if se])}\n"
                "Generate a concise, empathetic response for a medical chatbot."
            )
        else:
            prompt = f"Query: '{query}'\nNo diseases found for symptom '{symptom}'. Suggest asking for more details."
    elif intent == "treatment":
        doc = nlp(query.lower())
        disease = next((token.text for token in doc if token.pos_ == "NOUN" and token.dep_ in ["dobj", "nsubj"]), None)
        if disease:
            treatments = rag.get_treatment(disease)  # No extra quotes
            if treatments:
                prompt = (
                    f"Query: '{query}'\n"
                    f"Disease: {disease}\n"
                    f"Treatments: {', '.join(treatments)}\n"
                    "Provide a helpful treatment suggestion."
                )
    if not prompt:
        prompt = f"Query: '{query}'\nNo specific info found. Offer general assistance."

    prompt += "\nFormat response as: 'Selected Question: <question>' on first line, 'Humanized Answer: <response>' on second."
    selected_q, answer = llm.create_completion(prompt)
    return f"Selected Question: {selected_q or query}\nHumanized Answer: {answer}"

if __name__ == "__main__":
    initialize_knowledge_graph(metta)
    rag = MedicalRAG(metta)
    llm = GrokChat(api_key= os.environ.get("GROQ_API_KEY"))
    test_queries = [
            "Hi",
            "I have a fever",
            "I feel fatigue",
            "What’s wrong with me?",
            "How do I treat a migraine?",
            "What’s wrong with me if I have a cough?",
            "How do I treat depression?",
            "I have a headache",
            "How do I treat flu?",
            "What’s wrong with me if I have a headache?",
            "how can i treat flu?",
            "I feel anxiety",
    ]
    print("Welcome to the Medical Chatbot! Type 'exit' to quit.")

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        response = chatbot(user_input, rag, llm)
        print(response)  
 