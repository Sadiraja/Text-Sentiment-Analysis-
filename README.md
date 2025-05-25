# Text-Sentiment-Analysis


This project implements a simple **sentiment analysis system** on the IMDB movie reviews dataset using **TF-IDF vectorization** and a **logistic regression classifier built from scratch**. It classifies reviews as either *positive* or *negative* based on their content.

---

## 🔍 Features

- Custom text preprocessing: tokenization, stopword removal, and basic lemmatization
- TF-IDF feature extraction with minimum document frequency filtering
- Sparse matrix representation for efficiency
- Logistic regression classifier (no external ML libraries)
- Evaluation using Precision, Recall, and F1-score
- Real-time sentiment prediction for custom input

---

## 📁 Dataset

This project expects the dataset in CSV format with the following columns:
- `review`: the movie review text
- `sentiment`: the sentiment label (`positive` or `negative`)

> **Download IMDB Dataset:**  
> You can use the [IMDB Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/) or any CSV-formatted version named `imdb_dataset.csv`.

---

## 📦 Dependencies

- Python 3.6+
- pandas
- numpy
- scipy

Install them using:

```bash
pip install pandas numpy scipy
