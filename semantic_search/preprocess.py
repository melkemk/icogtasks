import json
import pandas as pd

# Load Kaggle dataset
with open('intents.json') as f:
    data = json.load(f)

# Convert to pandas DataFrame
rows = []
for intent in data['intents']:
    for pattern in intent['patterns']:
        rows.append({
            'question': pattern,
            'answer': intent['responses'][0]  # Take first response
        })

df = pd.DataFrame(rows)
df.to_csv('data/mental_health_faqs.csv', index=False)