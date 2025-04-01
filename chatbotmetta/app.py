import os
from flask import Flask, request, jsonify
from knowledge import MedicalRAG, metta
from chatbot import GrokChat, chatbot

app = Flask(__name__) 

# Initialize RAG and LLM 
rag = MedicalRAG(metta)
llm = GrokChat(api_key= os.environ.get("GROQ_API_KEY"))
@app.route("/ask-medical", methods=["GET"])
def ask_medical_get():
    """Handle GET requests with a query parameter."""
    query = request.args.get("question")
    if not query:
        return jsonify({"error": "No question provided"}), 400
    response = chatbot(query, rag, llm) 
    return jsonify({"answer": response})

@app.route("/ask-medical", methods=["POST"]) 
def ask_medical_post():  
    """Handle POST requests with JSON body."""
    data = request.get_json()
    if not data or "message" not in data or "text" not in data["message"]:
        return jsonify({"error": "Invalid request format. Use {'message': {'text': 'query'}}"}), 400
    
    query = data["message"]["text"] 
    if query.startswith("/add"):
        symptom_disease = llm.extract_medical_info(query[4:].strip())
        if len(symptom_disease) == 2 and symptom_disease[0] != "unknown" and symptom_disease[1] != "unknown":
            result = rag.add_knowledge("symptom", symptom_disease[0], symptom_disease[1])
            return jsonify({"status": "ok", "result": result})
        return jsonify({"status": "error", "result": "Failed to extract valid symptom-disease pair"})
    
    response = chatbot(query, rag, llm)
    return jsonify({"status": "ok", "answer": response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)