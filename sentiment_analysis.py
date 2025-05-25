import re
import math
import pandas as pd
import random
from collections import Counter, defaultdict
import numpy as np
from scipy.sparse import csr_matrix

# Custom stopword list (subset for brevity)
STOPWORDS = {'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was', 'were', 'will', 'with'}

# Simple lemmatization (rule-based for common cases)
def lemmatize(word):
    word = word.lower()
    if word.endswith('ing'):
        return word[:-3]
    if word.endswith('ed'):
        return word[:-2]
    if word.endswith('es'):
        return word[:-2]
    if word.endswith('s'):
        return word[:-1]
    return word

# Step 1: Text Preprocessing
def preprocess_text(text):
    tokens = re.findall(r'\b\w+\b', text.lower())
    tokens = [lemmatize(token) for token in tokens if token not in STOPWORDS and len(token) > 2]
    return tokens

# Step 2: Feature Engineering (TF-IDF)
def compute_tf(doc):
    tf = Counter(doc)
    total_terms = len(doc)
    return {term: count / total_terms for term, count in tf.items()}

def compute_idf(docs, min_df=5):
    doc_count = len(docs)
    term_doc_count = defaultdict(int)
    for doc in docs:
        unique_terms = set(doc)
        for term in unique_terms:
            term_doc_count[term] += 1
    return {term: math.log(doc_count / (1 + count)) for term, count in term_doc_count.items() if count >= min_df}

def compute_tfidf(docs, max_vocab_size=10000):
    idf = compute_idf(docs)
    tfidf_docs = []
    vocab = Counter()
    for doc in docs:
        tf = compute_tf(doc)
        tfidf = {term: tf_val * idf.get(term, 0) for term, tf_val in tf.items() if term in idf}
        tfidf_docs.append(tfidf)
        vocab.update(tf.keys())
    vocab = [term for term, _ in vocab.most_common(max_vocab_size)]
    return tfidf_docs, vocab

# Convert TF-IDF dicts to sparse matrix
def dict_to_vector(docs, vocab):
    row, col, data = [], [], []
    vocab_index = {term: idx for idx, term in enumerate(vocab)}
    for doc_idx, doc in enumerate(docs):
        for term, value in doc.items():
            if term in vocab_index:
                row.append(doc_idx)
                col.append(vocab_index[term])
                data.append(value)
    return csr_matrix((data, (row, col)), shape=(len(docs), len(vocab)))

# Step 3: Logistic Regression Classifier (from scratch, adapted for sparse matrices)
class LogisticRegression:
    def __init__(self, learning_rate=0.01, max_iter=100):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.weights = None
        self.bias = 0

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def train(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        for _ in range(self.max_iter):
            linear = X.dot(self.weights) + self.bias
            y_pred = self.sigmoid(linear)
            dw = (1 / n_samples) * X.T.dot(y_pred - y)
            db = (1 / n_samples) * np.sum(y_pred - y)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        linear = X.dot(self.weights) + self.bias
        y_pred = self.sigmoid(linear)
        return ['positive' if p >= 0.5 else 'negative' for p in y_pred]

# Step 4: Evaluation Metrics
def evaluate_model(true_labels, pred_labels):
    tp = tn = fp = fn = 0
    for true, pred in zip(true_labels, pred_labels):
        if true == 'positive' and pred == 'positive':
            tp += 1
        elif true == 'negative' and pred == 'negative':
            tn += 1
        elif true == 'positive' and pred == 'negative':
            fn += 1
        elif true == 'negative' and pred == 'positive':
            fp += 1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1

# Main function
def main():
    # Load IMDB dataset
    df = pd.read_csv('imdb_dataset.csv')  # Update with actual path
    reviews = df['review'].tolist()
    labels = df['sentiment'].tolist()

    # Preprocess texts
    processed_docs = [preprocess_text(review) for review in reviews]

    # Compute TF-IDF and get vocabulary
    tfidf_docs, vocab = compute_tfidf(processed_docs, max_vocab_size=10000)

    # Convert labels to binary (positive=1, negative=0)
    y = np.array([1 if label == 'positive' else 0 for label in labels])

    # Randomly shuffle data
    combined = list(zip(tfidf_docs, y))
    random.shuffle(combined)
    tfidf_docs, y = zip(*combined)
    tfidf_docs, y = list(tfidf_docs), np.array(list(y))

    # Split data (80% train, 20% test)
    split_idx = int(0.8 * len(tfidf_docs))
    train_docs, test_docs = tfidf_docs[:split_idx], tfidf_docs[split_idx:]
    train_y, test_y = y[:split_idx], y[split_idx:]
    test_labels = ['positive' if y == 1 else 'negative' for y in test_y]

    # Convert to sparse vectors
    X_train = dict_to_vector(train_docs, vocab)
    X_test = dict_to_vector(test_docs, vocab)

    # Train model
    classifier = LogisticRegression(learning_rate=0.1, max_iter=200)
    classifier.train(X_train, train_y)

    # Predict on test set
    predictions = classifier.predict(X_test)

    # Evaluate
    precision, recall, f1 = evaluate_model(test_labels, predictions)
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-Score: {f1:.2f}")

    # Multiple sample predictions
    samples = [
        "This movie was fantastic and really enjoyable!",
        "The plot was boring and the acting was terrible.",
        "An average film with some good moments."
    ]
    sample=input("Enter a sample text for sentiment analysis: ")

    idf = compute_idf(processed_docs)
    
    processed_sample = preprocess_text(sample)
    tf = compute_tf(processed_sample)
    tfidf_sample = {term: tf_val * idf.get(term, 0) for term, tf_val in tf.items()}
    X_sample = dict_to_vector([tfidf_sample], vocab)
    prediction = classifier.predict(X_sample)[0]
    print(f"\nSample text: {sample}")
    print(f"Predicted sentiment: {prediction}")

if __name__ == "__main__":
    main()