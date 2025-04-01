# Medical Chatbot

A Flask-based medical chatbot leveraging MeTTa for knowledge representation, spaCy for natural language processing, and Grok (via xAI) for humanized responses. This chatbot answers health-related questions about symptoms, diseases, treatments, and FAQs, drawing inspiration from reliable sources like MedlinePlus.

## Features

- **Symptom Queries**: Identify possible diseases from symptoms (e.g., "I have a fever" → flu).  
- **Treatment Suggestions**: Provide treatment options for diseases (e.g., "How do I treat flu?" → rest, fluids).  
- **Side Effects**: List potential side effects of treatments.  
- **FAQs**: Answer common questions sourced from MedlinePlus (e.g., "What are the symptoms of the flu?").  
- **Dynamic Knowledge**: Add new symptom-disease relationships via `/add` command.  
- **Keyword-Based Search**: Improved intent detection with predefined symptom, disease, and treatment keyword lists.  

## Project Structure

```
medical_chatbot/
├── app.py              # Flask app with API endpoints (ask_medical, ask_medical_message)
├── knowledge.py        # MeTTa knowledge graph and MedicalRAG class
├── chatbot.py          # Intent detection and response generation logic
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

## Prerequisites

- Python 3.8+  
- A Grok API key from xAI (set as an environment variable `GROQ_API_KEY`)  
- Postman (optional, for testing API endpoints)  

## Setup

### Clone the Repository:

### Create a Virtual Environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Install Dependencies:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_md
```

### Set Environment Variable:
Export your Grok API key:
```bash
export GROQ_API_KEY="your-grok-api-key-here"
```
Or add it to a `.env` file and use `python-dotenv` (optional, requires updating `app.py`).

### Run the App:
```bash
python app.py
```
The app will start on `http://localhost:8000` with debug mode enabled.

## Usage

The chatbot exposes two endpoints: `/ask-medical` (GET) and `/ask-medical` (POST).

### GET Request
- **Endpoint**: `http://localhost:8000/ask-medical`  
- **Parameter**: `question=<query>`  

#### Example:
```bash
curl "http://localhost:8000/ask-medical?question=What%20are%20the%20symptoms%20of%20the%20flu%3F"
```

#### Response:
```json
{
    "answer": "Selected Question: What are the symptoms of the flu?\nHumanized Answer: Flu symptoms include fever, cough, sore throat, body aches, and fatigue. Hope you’re not feeling too under the weather!"
}
```

### POST Request
- **Endpoint**: `http://localhost:8000/ask-medical`  
- **Body**: JSON with `{"message": {"text": "<query>"}}`  
- **Headers**: `Content-Type: application/json`  

#### Examples:
**Query:**
```bash
curl -X POST -H "Content-Type: application/json" -d '{"message": {"text": "I have a fever"}}' http://localhost:8000/ask-medical
```

**Response:**
```json
{
    "status": "ok",
    "answer": "Selected Question: I have a fever\nHumanized Answer: Oh no, a fever? That could be the flu—try resting and staying hydrated. Antiviral drugs might help if a doctor agrees. Take care!"
}
```

**Add Knowledge:**
```bash
curl -X POST -H "Content-Type: application/json" -d '{"message": {"text": "/add I feel dizzy"}}' http://localhost:8000/ask-medical
```

**Response:**
```json
{
    "status": "ok",
    "result": "Added symptom: dizzy → vertigo"
}
```

## Testing with Postman

### GET Request:
- **URL**: `http://localhost:8000/ask-medical?question=How%20do%20I%20treat%20flu%3F`  
- **Method**: GET  

Send and check the JSON response.

### POST Request:
- **URL**: `http://localhost:8000/ask-medical`  
- **Method**: POST  
- **Headers**: `Content-Type: application/json`  
- **Body (raw JSON)**:
```json
{
    "message": {
        "text": "What is the symptom of migraine"
    }
}
```

Send and verify the response.

## How It Works

- **Knowledge Base**: Stored in MeTTa (via `knowledge.py`), with symptoms, diseases, treatments, side effects, and FAQs inspired by MedlinePlus.  
- **Intent Detection**: Uses spaCy in `chatbot.py` to classify queries into "greeting," "symptom," "treatment," "diagnosis," or "faq."  
- **Keyword Extraction**: Matches query words against predefined lists (symptoms, diseases, treatments) for accurate MeTTa queries.  
- **Response Generation**: Grok humanizes responses based on MeTTa data or fallback prompts.  

## Example Queries

- "What are the symptoms of the flu?"  
- "I have a headache"  
- "How do I treat depression?"  
- "/add I feel anxious"  

## Troubleshooting

- **500 Errors**: Check terminal logs for MeTTa syntax issues or Grok API failures. Ensure `GROQ_API_KEY` is set.  
- **No Response**: Verify Flask is running on port 8000 and Postman/curl is hitting the correct URL.  
- **Missing Dependencies**: Run `pip install -r requirements.txt` again if modules are missing.  

## Future Improvements

- Add a web frontend for user-friendly interaction.  
- Enhance intent detection with machine learning models.  
- Expand the knowledge base with more medical data.  
- Add multilingual support.  
- Improve error handling and logging.  
- Integrate with voice assistants for accessibility.  
- Add user authentication for personalized responses.  
