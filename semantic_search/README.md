# Semantic Search Project

## Overview
This project implements a semantic search engine that leverages natural language processing (NLP) techniques to provide more accurate and relevant search results. Currently, the project uses a dataset of fewer than 500 entries, which may limit its performance. Future updates will focus on expanding the dataset and improving the search algorithms to enhance accuracy and relevance.

## Features
- **Semantic Understanding**: Interprets query meaning and context using Sentence-BERT embeddings.
- **Intent Recognition**: Detects greetings, farewells, and crisis-related queries for tailored responses.
- **Knowledge Graph**: Expands queries with related mental health concepts (e.g., "stress" → "anxiety").
- **Recommendations**: Offers practical suggestions like "Try meditation" for relevant queries.
- **Scalable Backend**: Built to handle growing datasets with a modular design.

## Screenshots
![Screenshot 1](s1.png)
![Screenshot 2](s2.png)

## Installation
To install the necessary dependencies, run:
```bash
pip install -r requirements.txt
```

## Usage
To start the semantic search engine, run:
```bash
docker-compose up
```
You will find the frontend at [http://localhost:3000](http://localhost:3000).

