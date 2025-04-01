from hyperon import MeTTa, E, S, ValueAtom

metta = MeTTa()

def initialize_knowledge_graph(metta):
    """Initialize the MeTTa knowledge graph with medical data."""
    # Symptoms → Diseases
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

    # Treatments → Side Effects
    metta.space().add_atom(E(S("side_effect"), S("antiviral drugs"), ValueAtom("nausea, dizziness")))
    metta.space().add_atom(E(S("side_effect"), S("pain relievers"), ValueAtom("stomach upset")))
    metta.space().add_atom(E(S("side_effect"), S("antidepressants"), ValueAtom("weight gain, insomnia")))
    metta.space().add_atom(E(S("side_effect"), S("therapy"), ValueAtom("initial discomfort")))
    metta.space().add_atom(E(S("side_effect"), S("medications"), ValueAtom("drowsiness, dry mouth")))

    # FAQs (expanded from MedlinePlus, https://medlineplus.gov)
    metta.space().add_atom(E(S("faq"), S("Hi"), ValueAtom("Hello! How can I assist you with your health today?")))
    metta.space().add_atom(E(S("faq"), S("What’s wrong with me?"), ValueAtom("I’m not a doctor, but I can help explore your symptoms. What are you feeling?")))
    metta.space().add_atom(E(S("faq"), S("How do I treat a migraine?"), ValueAtom("For migraines, try resting in a dark room, staying hydrated, and using pain relievers.")))
    metta.space().add_atom(E(S("faq"), S("How do I treat flu?"), ValueAtom("For the flu, rest, drink fluids, and consider antiviral drugs if prescribed.")))
    metta.space().add_atom(E(S("faq"), S("What are the symptoms of the flu?"), ValueAtom("Flu symptoms include fever, cough, sore throat, body aches, and fatigue.")))
    metta.space().add_atom(E(S("faq"), S("When should I see a doctor for a fever?"), ValueAtom("See a doctor if your fever is 103°F or higher, lasts over 3 days, or includes severe symptoms like breathing trouble.")))

class MedicalRAG:
    def __init__(self, metta_instance):
        self.metta = metta_instance
        initialize_knowledge_graph(self.metta)

    def query_symptom(self, symptom):
        """Find diseases linked to a symptom."""
        symptom = symptom.strip('"')
        print(symptom,'symptom')
        query_str = f'!(match &self (symptom {symptom} $disease) $disease)'
        results = self.metta.run(query_str)
        return list(set(str(r[0]) for r in results if r and len(r) > 0)) if results else []

    def get_treatment(self, disease):
        """Find treatments for a disease."""
        disease = disease.strip('"')
        print(disease,'disease')
        query_str = f'!(match &self (treatment {disease} $treatment) $treatment)'
        results = self.metta.run(query_str)
        return [r[0].get_object().value for r in results if r and len(r) > 0] if results else []

    def get_side_effects(self, treatment):
        """Find side effects of a treatment."""
        treatment = treatment.strip('"')
        query_str = f'!(match &self (side_effect {treatment} $effect) $effect)'
        results = self.metta.run(query_str)
        return [r[0].get_object().value for r in results if r and len(r) > 0] if results else []

    def query_faq(self, question):
        """Retrieve FAQ answers."""
        query_str = f'!(match &self (faq "{question}") $answer)'
        results = self.metta.run(query_str)
        return results[0][0].get_object().value if results and results[0] else None

    def add_knowledge(self, relation_type, subject, object_value):
        """Add new knowledge dynamically."""
        if isinstance(object_value, str):
            object_value = ValueAtom(object_value)
        self.metta.space().add_atom(E(S(relation_type), S(subject), object_value))
        return f"Added {relation_type}: {subject} → {object_value}" 