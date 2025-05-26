import numpy as np
from collections import defaultdict
import pickle

class EmbeddingLayer:
    def __init__(self, sentences, embedding_dim=100):
        self.sentences = sentences
        self.word2idx = dict()
        self.idx2word = dict()
        self.length = 0
        self.embedding_dim = embedding_dim
        self.embeddings = None
        self.word_counts = defaultdict(int)
        
    def get_vocab(self):
        i = 1
        temp = set()
        for sentence in self.sentences:
            for word in sentence.split():
                self.word_counts[word] += 1
                if word not in temp:
                    temp.add(word)
                    self.word2idx[word] = i
                    self.idx2word[i] = word
                    i += 1
        
        # Add padding token at index 0
        self.word2idx['<PAD>'] = 0
        self.idx2word[0] = '<PAD>'
        
        self.length = i
        return i, self.word2idx, self.idx2word
    
    def prev_words(self, i, doc, window_size):
        out = []
        for index in range(i-window_size, i):
            if index >= 0:
                out.append(self.word2idx[doc[index]])
            else:
                out.append(0)
        return out
    
    def next_words(self, i, doc, window_size):
        out = []
        for index in range(i+1, i+window_size+1, 1):
            if index < len(doc):
                out.append(self.word2idx[doc[index]])
            else:
                out.append(0)
        return out
    
    def get_training_data(self, sentence=None, window_size=6):
        X = []
        y = []
        for sentence in self.sentences:
            xi = []
            yi = []
            sentence = sentence.split()
            for index, word in enumerate(sentence):
                prev = self.prev_words(index, sentence, window_size)
                next = self.next_words(index, sentence, window_size)
                assert len(prev) == len(next)
                xi.append(prev + next)
                yi.append(self.word2idx[word])
            X.extend(xi)
            y.extend(yi)
        return X, y
    
    def initialize_embeddings(self):
        """Initialize embedding matrix with random values"""
        self.embeddings = np.random.normal(0, 0.1, (self.length, self.embedding_dim))
        return self.embeddings
    
    def get_word_vector(self, word):
        """Get embedding vector for a specific word"""
        if word in self.word2idx:
            idx = self.word2idx[word]
            return self.embeddings[idx] if self.embeddings is not None else None
        else:
            print(f"Word '{word}' not found in vocabulary")
            return None
    
    def get_vectors_for_context(self, context_indices):
        """Get average of context word vectors"""
        if self.embeddings is None:
            print("Embeddings not initialized")
            return None
        
        valid_indices = [idx for idx in context_indices if idx != 0]  # Remove padding
        if not valid_indices:
            return np.zeros(self.embedding_dim)
        
        context_vectors = self.embeddings[valid_indices]
        return np.mean(context_vectors, axis=0)
    
    def cosine_similarity(self, word1, word2):
        """Calculate cosine similarity between two words"""
        vec1 = self.get_word_vector(word1)
        vec2 = self.get_word_vector(word2)
        
        if vec1 is None or vec2 is None:
            return None
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0
        
        return dot_product / (norm1 * norm2)
    
    def find_similar_words(self, word, top_k=5):
        """Find most similar words to given word"""
        if word not in self.word2idx:
            print(f"Word '{word}' not found in vocabulary")
            return []
        
        if self.embeddings is None:
            print("Embeddings not initialized")
            return []
        
        similarities = {}
        target_vector = self.get_word_vector(word)
        
        for other_word in self.word2idx:
            if other_word != word and other_word != '<PAD>':
                sim = self.cosine_similarity(word, other_word)
                if sim is not None:
                    similarities[other_word] = sim
        
        # Sort by similarity and return top k
        sorted_words = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        return sorted_words[:top_k]
    
    def update_embeddings(self, word_idx, gradient, learning_rate=0.01):
        """Update embedding for a specific word"""
        if self.embeddings is not None and 0 <= word_idx < len(self.embeddings):
            self.embeddings[word_idx] -= learning_rate * gradient
    
    def get_word_frequency(self, word):
        """Get frequency of a word in the corpus"""
        return self.word_counts.get(word, 0)
    
    def get_vocab_stats(self):
        """Get vocabulary statistics"""
        total_words = sum(self.word_counts.values())
        unique_words = len(self.word2idx) - 1  # Exclude padding token
        
        return {
            'total_words': total_words,
            'unique_words': unique_words,
            'vocab_size': self.length,
            'embedding_dim': self.embedding_dim
        }
    
    def save_embeddings(self, filepath):
        """Save embeddings to file"""
        embedding_data = {
            'embeddings': self.embeddings,
            'word2idx': self.word2idx,
            'idx2word': self.idx2word,
            'embedding_dim': self.embedding_dim,
            'vocab_size': self.length
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(embedding_data, f)
        print(f"Embeddings saved to {filepath}")
    
    def load_embeddings(self, filepath):
        """Load embeddings from file"""
        try:
            with open(filepath, 'rb') as f:
                embedding_data = pickle.load(f)
            
            self.embeddings = embedding_data['embeddings']
            self.word2idx = embedding_data['word2idx']
            self.idx2word = embedding_data['idx2word']
            self.embedding_dim = embedding_data['embedding_dim']
            self.length = embedding_data['vocab_size']
            
            print(f"Embeddings loaded from {filepath}")
            return True
        except Exception as e:
            print(f"Error loading embeddings: {e}")
            return False
    
    def sentence_to_indices(self, sentence):
        """Convert sentence to list of word indices"""
        words = sentence.split()
        indices = []
        for word in words:
            if word in self.word2idx:
                indices.append(self.word2idx[word])
            else:
                indices.append(0)  # Use padding for unknown words
        return indices
    
    def indices_to_sentence(self, indices):
        """Convert list of indices back to sentence"""
        words = []
        for idx in indices:
            if idx in self.idx2word and idx != 0:
                words.append(self.idx2word[idx])
        return ' '.join(words)


if __name__ == "__main__":
    # Testing the extended embedding layer
    sentences = [
        'edbert is nice',
        'vanson is handsome',
        'zaki is gay',
        'nice people are handsome',
        'handsome people are nice'
    ]
    
    EL = EmbeddingLayer(sentences=sentences, embedding_dim=3)
    
    # Get vocabulary
    vocab_size, word2idx, idx2word = EL.get_vocab()
    print(f"Vocabulary size: {vocab_size}")
    print(f"Word to index mapping: {word2idx}")
    print(f"Index to word mapping: {idx2word}")
    
    # Get training data
    X, y = EL.get_training_data(window_size=2)
    print(f"\nTraining data shape - X: {len(X)}, y: {len(y)}")
    print(f"Sample context: {X[0]} -> target: {y[0]}")
    
    # Initialize embeddings
    embeddings = EL.initialize_embeddings()
    print(f"\nEmbedding matrix shape: {embeddings.shape}")
    
    # Test word vector retrieval
    word_vec = EL.get_word_vector('nice')
    print(f"\nVector for 'nice': {word_vec[:5]}...")  # Show first 5 dimensions
    
    # Test similarity (with random embeddings, similarities won't be meaningful)
    sim = EL.cosine_similarity('nice', 'handsome')
    print(f"\nCosine similarity between 'nice' and 'handsome': {sim}")
    
    # Get vocabulary statistics
    stats = EL.get_vocab_stats()
    print(f"\nVocabulary statistics: {stats}")
    
    # Test sentence conversion
    test_sentence = "nice people"
    indices = EL.sentence_to_indices(test_sentence)
    reconstructed = EL.indices_to_sentence(indices)
    print(f"\nOriginal: '{test_sentence}' -> Indices: {indices} -> Reconstructed: '{reconstructed}'")