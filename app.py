import pandas as pd
import joblib
import string
from nltk.corpus import stopwords


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



# load trained model
model = joblib.load(

    "model/emotion_model.pkl"
)


# load songs dataset
songs = pd.read_csv(

    "songs/songs_by_emotion.csv"
)


def recommend_song(emotion):

    filtered = songs[

        songs["emotion"] == emotion
    ]

    row = filtered.sample(1)

    return row.iloc[0]["song"], row.iloc[0]["artist"]



print("\nEmotion Based Song Recommendation System")

print("-----------------------------------------")


while True:

    text = input("\nEnter your feeling: ")

    clean = clean_text(text)

    emotion = model.predict([clean])[0]

    song, artist = recommend_song(emotion)

    print("\nDetected Emotion →", emotion)

    print("Recommended Song →", song)

    print("Artist →", artist)

    again = input("\nTry again? yes/no: ")

    if again == "no":

        break
