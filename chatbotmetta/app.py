# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
from hyperon import MeTTa
from knowledge import initialize_knowledge_graph
from medicalrag import MedicalRAG
from chatbot import GrokChat, chatbot
import threading

# Load environment variables
load_dotenv()

app = Flask(__name__)
# Enable CORS
CORS(app)


metta = MeTTa()
initialize_knowledge_graph(metta)
rag = MedicalRAG(metta)
llm = GrokChat(api_key=os.getenv("GROQ_API_KEY"))

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    if not data or "query" not in data:
        return jsonify({"error": "Missing 'query' in request body"}), 400
    
    query = data["query"]
    response = chatbot(query, rag, llm)
    return jsonify(response)

def run_server():
    app.run(debug=True, host="0.0.0.0", port=5000, use_reloader=False)

def interactive_chat():
    print('interactive chat')
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        response = chatbot(user_input, rag, llm)
        print(response)

if __name__ == "__main__":
    # Run the Flask server in a separate thread
    server_thread = threading.Thread(target=run_server)
    server_thread.daemon = True
    server_thread.start()

    interactive_chat()