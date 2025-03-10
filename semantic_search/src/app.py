from flask import Flask, request, jsonify
from search_engine import MentalHealthSearch
from flask_cors import CORS  # Import CORS

app = Flask(__name__)
search_engine = MentalHealthSearch()
CORS(app) 
@app.route('/search', methods=['POST'])
def search():
    print(1111)
    data = request.json
    query = data.get('query', "")
    context = data.get('context', [])
    print(context)
    
    # Directly get the search results
    result = search_engine.search(query, context)
    
    # Convert tensor to list for JSON serialization
    result['scores'] = result['scores'].tolist()
    print(result,'what is this')
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
