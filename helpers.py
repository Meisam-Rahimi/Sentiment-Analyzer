import nltk
import string
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
import shutil
import textwrap
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import random


nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)


def print_wrapped_text(text):
    # Get the width of the terminal window
    terminal_width = shutil.get_terminal_size().columns

    # Wrap the text
    wrapped_text = textwrap.fill(text, width=terminal_width)

    # Print the wrapped text
    print(wrapped_text)


def get_wordnet_pos(tag):
    # Convert NLTK POS tags to WordNet POS tags
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # Default to noun


def preprocess(text):
    # Lowercase
    text = text.lower()

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords and punctuation
    tokens = [word for word in tokens if word not in stopwords.words('english') and word not in string.punctuation]

    # POS tagging
    tagged_tokens = pos_tag(tokens)

    # Lemmatization with POS
    lemmatizer = WordNetLemmatizer()
    lemmatized = [
        lemmatizer.lemmatize(word, get_wordnet_pos(pos_tag))
        for word, pos_tag in tagged_tokens
    ]

    return " ".join(lemmatized)


# Load dataset, preprocess, vectorize, train
def train_model():
    # Load the CSV file
    df = pd.read_csv('train_test.csv')

    # Apply preprocessing to the review column
    df['clean_text'] = df['review'].apply(preprocess)

    # Encode sentiment labels (positive → 1, negative → 0)
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['sentiment'])

    # Extract reviews and sentiment labels from the dataframe
    X_text = df['clean_text']
    y = df['label']

    # Vectorize the preprocessed reviews
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(X_text)

    # Split data into test and train sets (20 percent of data for training) and train the logistic regression model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Measure the accuracy of the model
    accuracy = model.score(X_test, y_test)
    print(f"Accuracy: {accuracy:.2f}")
    print("")

    return model, vectorizer, label_encoder


# Select a random review, show the review and its corresponding sentiment, and predict the sentiment using the model
def predict_random_review(model, vectorizer, label_encoder):

    # Pick a random index between 0 and 49000, the number of reviews in the database
    random_index = random.randint(0, 49000)
    # Select the review in that randomly selected index and show the review and its corresponding sentiment
    df_random = pd.read_csv('database.csv')
    random_review = df_random.loc[random_index, 'review']
    random_sentiment = df_random.loc[random_index, 'sentiment']
    print("The random review selected is:")
    print_wrapped_text(random_review)
    print("")
    print("The corresponding sentiment determined in the database is: ", random_sentiment)
    print("")
    # Predict the sentiment of the randomly selected review to compare it with the predetermined sentiment
    clean_random = preprocess(random_review)
    random_vector = vectorizer.transform([clean_random])
    prediction = model.predict(random_vector)
    print("The predicted sentiment for the randomly selected review is: ", label_encoder.inverse_transform(prediction)[0])
    print("")


def predict_sentiment(model, vectorizer, label_encoder, user_input):
    # Preprocess the input
    clean_input = preprocess(user_input)
    # Vectorize it
    input_vector = vectorizer.transform([clean_input])
    # Predict
    prediction = model.predict(input_vector)
    # Decode prediction
    sentiment = label_encoder.inverse_transform(prediction)[0]

    return sentiment