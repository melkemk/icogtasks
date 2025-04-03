import token


class SimpleWordTokenizer:
    def __init__(self, corpus):
        self.word2id = {}
        self.id2word = {}
        self.build_vocabulary(corpus)
    
    def build_vocabulary(self, corpus):
        # Collect unique words
        unique_words = set()
        for sentence in corpus:
            words = sentence.lower().split()  
            unique_words.update(words)
        
        self.word2id["<UNK>"] = 0  
        unique_words = list(sorted(unique_words)) 
        self.word2id = {word:i+1 for i,word in enumerate(unique_words)}
        # print(unique_wordswithindex)
        # self.word2id = {word:i+1 for i,word in (unique_wordswithindex)  } 
        # self.word2id[word] = i
        # print(self.word2id)
        self.id2word = {i: w for w, i in self.word2id.items()}
        print(f"Vocabulary size: {len(self.word2id)}")
    
    def encode(self, text): 
        # Convert text to token IDs
        words = text.lower().split()
        return [self.word2id.get(word, 0) for word in words]  # Use <UNK> for OOV
    
    def decode(self, token_ids):
        print(token_ids)
        # Convert IDs back to text
        return " ".join(self.id2word.get(id, "<UNK>") for id in token_ids)

# Example usage
corpus = [
    "the cat sat on the mat",
    "the dog jumped over the cat",
    "a quick brown fox"
]
tokenizer = SimpleWordTokenizer(corpus)

text = "the cat jumped what"
encoded = tokenizer.encode(text)
decoded = tokenizer.decode(encoded)
print(f"Text: {text}")
print(f"Encoded: {encoded}")
print(f"Decoded: {decoded}") 