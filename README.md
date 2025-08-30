# README.md

# Filtering the Noise: ML for Trustworthy Location Reviews

---

## Project Overview

This Jupyter Notebook pipeline allows you to clean, preprocess, and classify online reviews. The workflow includes:

1. Downloading Kaggle datasets and public JSON review files.
2. Cleaning and preprocessing text:
   - Converting text to lowercase
   - Removing punctuation, numbers, and special characters
   - Removing stopwords and lemmatizing words
3. Sampling reviews for testing or using full dataset.
4. Renaming columns for clarity:
   - `business_name` → `location_name`
   - `text` → `review_text`
   - `author_name` → `review_id`
5. Optional: Fetching Google store categories via Google Places API.
6. Classifying reviews using FLAN-T5 into:
   - Spam
   - Advertisement
   - Irrelevant
   - Rant/Fake Complaint
   - Genuine Review
7. Flagging policy violations using a RandomForest classifier.
8. Computing sentiment scores using VADER.
9. Calculating review quality score and ranking.
10. Saving the final processed dataset.

---

## Setup Instructions

1. Open Google Colab and ensure GPU is enabled (recommended for LLM inference).
2. Install required packages:

```python
!pip install transformers pandas requests tqdm scikit-learn torch nltk kagglehub pillow
```

3. Import packages and download NLTK resources:

```python
import pandas as pd
import requests
import nltk
import os
import gzip
import json
import random
import re
from tqdm import tqdm
from kagglehub import KaggleDatasetAdapter, kagglehub
from transformers import pipeline
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from scipy.sparse import hstack
from huggingface_hub import notebook_login

nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

notebook_login()
```

4. Download Kaggle dataset or public JSON reviews.

5. Clean and preprocess text:

```python
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [lemmatizer.lemmatize(t) for t in text.split() if t not in stop_words]
    return " ".join(tokens)
```

6. Rename columns for clarity:

```python
df.rename(columns={
    "business_name": "location_name",
    "text": "review_text",
    "author_name": "review_id"
}, inplace=True)
```

---

## How to Reproduce Results

1. Configure inputs and sampling options:

```python
INPUT_PATH = "master_csv.csv"
TEST_MODE = True
SAMPLE_SIZE = 100  # for testing, use full dataset if False
```

2. Load dataset:

```python
def load_reviews(path):
    import os, glob, gzip, json, pandas as pd
    if os.path.isfile(path) and path.endswith(".csv"):
        return pd.read_csv(path)
    elif os.path.isdir(path):
        all_reviews = []
        for f in glob.glob(f"{path}/*.json.gz"):
            with gzip.open(f, "rt", encoding="utf-8") as fh:
                for line in fh:
                    all_reviews.append(json.loads(line))
        return pd.DataFrame(all_reviews)
    else:
        raise ValueError(f"invalid path: {path}.")

df = load_reviews(INPUT_PATH)
```

3. Optional: Fetch Google store categories using API key.

4. Classify reviews using FLAN-T5 in batches.

5. Compute policy violations, sentiment scores, and quality ranking.

6. Save final dataset:

```python
OUTPUT_FILE = INPUT_PATH.replace(".csv", "_classified.csv")
df.to_csv(OUTPUT_FILE, index=False)
```

