import pandas as pd
import nltk
import string
import joblib

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# download stopwords
nltk.download('stopwords')

# load dataset
data = pd.read_csv("dataset/emotion_dataset.csv")

# clean text function
def clean_text(text):

    text = text.lower()

    text = "".join(
        char for char in text
        if char not in string.punctuation
    )

    words = text.split()

    words = [
        w for w in words
        if w not in stopwords.words("english")
    ]

    return " ".join(words)


# apply cleaning
data["clean_text"] = data["text"].apply(clean_text)

# ML pipeline
model = Pipeline([

    ("tfidf", TfidfVectorizer()),

    ("classifier", MultinomialNB())
])

# train model
model.fit(

    data["clean_text"],

    data["emotion"]
)

# save model
joblib.dump(

    model,

    "model/emotion_model.pkl"
)

print("Model trained successfully")
