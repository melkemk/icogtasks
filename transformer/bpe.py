import json
import os
import re
from collections import Counter

class BPETokenizer:
    def __init__(self, vocab_size=50):
        self.vocab_size = vocab_size
        self.vocab = set()  # Full set of tokens
        self.merges = []    # List of merge rules

    def train(self, corpus):
        # Split corpus into words and count frequencies
        words = corpus.split()
        word_freqs = Counter(words)

        # Initialize with characters (split each word into chars + </w>)
        vocab = {word: freq for word, freq in word_freqs.items()}
        tokenized_words = {word: list(word) + ['</w>'] for word in vocab}
# edge case on teh last end 
        # Add initial characters to vocab
        # print(tokenized_words)
        for word in tokenized_words.values():
            self.vocab.update(word)
        # print()
        # print(self.vocab)
        # Merge pairs until vocab_size is reached
        while len(self.vocab) < self.vocab_size:
            # Count pairs across all tokenized words
            pairs = Counter() 
            for word, freq in vocab.items(): 
                tokens = tokenized_words[word]
                for i in range(len(tokens) - 1):
                    pairs[(tokens[i], tokens[i + 1])] += freq

            if not pairs:
                print(1)
                break
            # Find most frequent pair
            best_pair = max(pairs, key=pairs.get)
            self.merges.append(best_pair)
            print(f"Merging {best_pair} , vocab size: {len(self.vocab)}")
            # Merge the best pair in all words
            for word in vocab: 
                tokens = tokenized_words[word]
                new_tokens = []
                i = 0
                while i < len(tokens):
                    if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best_pair:
                        new_tokens.append(tokens[i] + tokens[i + 1])
                        i += 2
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                tokenized_words[word] = new_tokens
                self.vocab.update(new_tokens)
                # print(self.vocab,111)


    def encode(self, text):
        # Split text into words and tokenize each into characters + </w>
        words = text.split()
        tokens = []
        for word in words: 
            word_tokens = list(word) + ['</w>']
            # Apply all merges in order
            for pair in self.merges:
                i = 0
                new_tokens = []
                while i < len(word_tokens):
                    if i < len(word_tokens) - 1 and (word_tokens[i], word_tokens[i + 1]) == pair:
                        new_tokens.append(word_tokens[i] + word_tokens[i + 1])
                        i += 2
                    else:
                        new_tokens.append(word_tokens[i])
                        i += 1
                word_tokens = new_tokens
            tokens.extend(word_tokens)
        return tokens 
    def decode(self, tokens):
        text = ""
        for token in tokens:
            if token.endswith("</w>"):
                text += token[:-4] + " "  # Remove '</w>' and add a space
            else:
                text += token
        return text.strip() 


train_file = './data/ptb.train.txt'
test_file = './data/ptb.test.txt'


# Check if files exist
if not os.path.exists(train_file):
    raise FileNotFoundError(f"Training file {train_file} not found!")
if not os.path.exists(test_file):
    raise FileNotFoundError(f"Test file {test_file} not found!")

# Load the full training and test data
with open(train_file, 'r', encoding='utf-8') as f:
    corpus = f.read() 
    
with open(test_file, 'r', encoding='utf-8') as f:
    test_sentence = f.read() 

# Set vocab_size to 8001 as per the paper
tokenizer = BPETokenizer(vocab_size=8001)
tokenizer.train(corpus)

# Save merges to a file
merges_file = 'merges.json'
with open(merges_file, 'w', encoding='utf-8') as f:
    json.dump(tokenizer.merges, f)
print(f"Merges saved to {merges_file}")

# Output results
print("Vocabulary size:", len(tokenizer.vocab)) 
print("Sample vocabulary:", sorted(list(tokenizer.vocab))[:20])  # Show first 20 tokens

# Encode a portion of the test file
test_sentence_chunk = test_sentence[:200]  # Limit for readability
encoded = tokenizer.encode(test_sentence_chunk)
print("Test sentence chunk:", test_sentence_chunk)
print("Encoded Text:", encoded)
decoded = tokenizer.decode(encoded)
print("Decoded Text:", decoded)
