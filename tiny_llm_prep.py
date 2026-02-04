import os
import pickle


class TinyTokenizer:
    def __init__(self, raw_text=None, load_path=None):
        """
        Initialize the tokenizer.
        Provide raw_text to build a new vocab,
        or load_path to load an existing one from your machine.
        """
        if load_path and os.path.exists(load_path):
            self.load_vocab(load_path)
        elif raw_text:
            # Clean and split the text into words/tokens
            self.tokens = raw_text.lower().split()
            self.unique_tokens = sorted(list(set(self.tokens)))
            self.vocab_size = len(self.unique_tokens)

            # Create the mapping dictionaries
            self.stoi = {word: i for i, word in enumerate(self.unique_tokens)}
            self.itos = {i: word for i, word in enumerate(self.unique_tokens)}
        else:
            print("Error: Provide either raw_text or a valid load_path.")

    def encode(self, s):
        words = s.lower().split()
        return [self.stoi[w] for w in words if w in self.stoi]

    def decode(self, l):
        return ' '.join([self.itos[i] for i in l])

    def save_vocab(self, filepath):
        """Saves the vocabulary to your local disk."""
        with open(filepath, 'wb') as f:
            pickle.dump({'stoi': self.stoi, 'itos': self.itos}, f)
        print(f"Vocab saved to {filepath}")

    def load_vocab(self, filepath):
        """Loads a previously saved vocabulary."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.stoi = data['stoi']
            self.itos = data['itos']
            self.vocab_size = len(self.stoi)
        print(f"Vocab loaded from {filepath}")


def load_local_data(directory_path):
    """
    Reads all .txt files in a local folder and merges them.
    """
    all_text = ""
    if not os.path.exists(directory_path):
        return "Once upon a time there was a small bird. It liked to fly high in the sky."  # Fallback sample

    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            with open(os.path.join(directory_path, filename), 'r', encoding='utf-8') as f:
                all_text += f.read() + " "
    return all_text


# --- LOCAL EXECUTION STEPS ---

# 1. Path to your TinyStories folder (Change this to your actual folder path)
DATA_DIR = "./data/tinystories"

# 2. Load the text from your machine
print("Loading local stories...")
raw_data = load_local_data(DATA_DIR)

# 3. Initialize Tokenizer
tokenizer = TinyTokenizer(raw_text=raw_data)
print(f"Successfully built local vocabulary. Size: {tokenizer.vocab_size}")

# 4. Save for later so we don't have to re-process every time
tokenizer.save_vocab("vocab.pkl")

# 5. Quick Test
test_str = "The bird flew high."
encoded = tokenizer.encode(test_str)
print(f"\nTest String: {test_str}")
print(f"Encoded IDs: {encoded}")
print(f"Decoded: {tokenizer.decode(encoded)}")