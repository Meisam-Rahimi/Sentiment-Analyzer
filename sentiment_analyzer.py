from helpers import *
import time


def main():
    print("Loading and training model...")
    # Records the start of train and test to measure train and test time
    start_time = time.time()

    model, vectorizer, label_encoder = train_model()
    print(f"Training complete in {time.time() - start_time:.2f} seconds")
    print("")

    predict_random_review(model, vectorizer, label_encoder)

    while True:
        user_input = input("Enter text for sentiment analysis (or press Enter to quit): ")
        if not user_input:
            break
        sentiment = predict_sentiment(model, vectorizer, label_encoder, user_input)
        print(f"Predicted Sentiment: {sentiment}\n")

if __name__ == "__main__":
    main()
