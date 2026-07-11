# Twitter Sentiment Analysis using Machine Learning

A machine learning project that classifies tweets as **Positive** or **Negative** using **TF-IDF Vectorization** and **Logistic Regression**.

The project uses the **Sentiment140** dataset from Kaggle, containing **1.6 million tweets**.

---

## Features

- Data preprocessing
- Text cleaning
- Stopword removal
- Porter Stemming
- TF-IDF Vectorization
- Logistic Regression classifier
- Model evaluation
- Model serialization using Pickle

---

## Dataset

Dataset: **Sentiment140**

https://www.kaggle.com/datasets/kazanova/sentiment140

Number of Tweets:
- 1,600,000

Classes:
- 0 → Negative
- 4 → Positive (converted to 1)

---

## Technologies Used

- Python
- Pandas
- NumPy
- NLTK
- Scikit-learn
- Kaggle API
- Google Colab

---

## Project Workflow

1. Download dataset using Kaggle API
2. Extract dataset
3. Load CSV using Pandas
4. Data preprocessing
    - Lowercase conversion
    - Remove special characters
    - Remove stopwords
    - Porter Stemming
5. Train-test split
6. TF-IDF Vectorization
7. Logistic Regression
8. Evaluate Accuracy
9. Save trained model using Pickle

---

## Installation

Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/twitter-sentiment-analysis.git

cd twitter-sentiment-analysis
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Kaggle API Setup

## Step 1

Create a Kaggle account.

https://www.kaggle.com

---

## Step 2

Generate your API Token.

Go to

```
Account → API → Create New Token
```

A file named

```
kaggle.json
```

will be downloaded.

---

## Google Colab Users

Instead of uploading the token, store your credentials securely using **Colab Secrets**.

Create two secrets:

| Secret Name | Value |
|-------------|-------|
| KAGGLE_USER | Your Kaggle Username |
| KAGGLE_SECRET | Your Kaggle API Key |

Then load them:

```python
from google.colab import userdata
import os

os.environ["KAGGLE_USERNAME"] = userdata.get("KAGGLE_USER")
os.environ["KAGGLE_KEY"] = userdata.get("KAGGLE_SECRET")
```

Download dataset:

```bash
!kaggle datasets download -d kazanova/sentiment140
```

Extract:

```python
from zipfile import ZipFile

with ZipFile("sentiment140.zip","r") as zip:
    zip.extractall()
```

---

## Local Machine Setup

Move the downloaded

```
kaggle.json
```

to

Linux / macOS

```
~/.kaggle/kaggle.json
```

Windows

```
C:\Users\<username>\.kaggle\kaggle.json
```

Set permissions (Linux):

```bash
chmod 600 ~/.kaggle/kaggle.json
```

Then download:

```bash
kaggle datasets download -d kazanova/sentiment140
```

---

## Results

Training Accuracy

```
~80%
```

Testing Accuracy

```
~77%
```

(Actual values may vary slightly.)

---

## Model

The trained model is saved as

```
trained_model.sav
```

Load the model

```python
import pickle

model = pickle.load(open("trained_model.sav","rb"))
```

---

## Future Improvements

- LSTM
- GRU
- BERT
- RoBERTa
- DistilBERT
- Hyperparameter tuning
- Flask/FastAPI deployment
- Streamlit Web App

---

## License

MIT License
