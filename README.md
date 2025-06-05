# Sentiment Analyzer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A simple yet effective sentiment analysis tool that classifies movie reviews as positive or negative. Built using Python and scikit-learn, the project uses a subset of the IMDB 50K Movie Reviews dataset and includes features such as text preprocessing, model evaluation, and interactive prediction.
________________________________________
📂 Dataset
The project uses the IMDB dataset containing 50,000 labeled movie reviews:
•	train_test.csv – A sample of 1,000 reviews used for training and testing the model.
•	database.csv – The remaining 49,000 reviews, used for additional evaluation and demonstration.
Due to hardware limitations, only a portion of the data was used for model training.
________________________________________
🔄 Text Preprocessing
Each review undergoes the following preprocessing steps:
•	Lowercasing all text
•	Removing punctuation
•	Tokenizing the text into individual words
•	Removing stopwords (e.g., "the", "is", "and")
•	Lemmatization (e.g., "running" → "run", "better" → "good")
These steps clean and normalize the text for improved classification performance.
________________________________________
🔢 Feature Extraction
The cleaned text is converted into numerical vectors using TF-IDF Vectorization from scikit-learn. This helps the model determine the importance of words relative to all reviews.
________________________________________
🧠 Model Training
A Logistic Regression model is trained using the vectorized text data:
•	Model training and testing times are recorded and displayed.
•	Accuracy is calculated and printed after training.
________________________________________
🧪 Model Evaluation
To evaluate model performance beyond the training set:
•	A random review is selected from database.csv.
•	The review's text and true sentiment are displayed.
•	The model then predicts the sentiment and outputs the result for comparison.
________________________________________
👤 Interactive Prediction
After evaluation, the script enters an interactive mode:
•	The user can input any custom text.
•	The model predicts and displays the sentiment (positive/negative).
•	The loop continues until the user presses Enter on an empty input.
________________________________________
🚀 How to Run
1.	Clone this repository:
2.	git clone https://github.com/yourusername/sentiment-analyzer.git
3.	cd sentiment-analyzer
4.	Install dependencies (optional virtual environment recommended):
5.	pip install -r requirements.txt
6.	Run the script:
7.	python sentiment_analyzer.py
________________________________________
📦 Dependencies
•	Python 3.8+
•	scikit-learn
•	pandas
•	nltk
________________________________________
📌 Notes
•	Lemmatization uses NLTK’s WordNetLemmatizer. Ensure required NLTK data is downloaded:
•	import nltk
•	nltk.download('punkt')
•	nltk.download('stopwords')
•	nltk.download('wordnet')
•	nltk.download('omw-1.4')
•	The model uses only a portion of the full dataset due to memory and CPU constraints.
________________________________________
📈 Example Output

Loading and training model...
Accuracy: 0.82

Training complete in 105.83 seconds

The random review selected is:
This movie is a real low budget production, yet I will not say anything more on 
that as it already has been covered. I give this movie a low rating for the story
 alone, but I met the director the night I saw the film and he gave me an additional
 reason to dislike the movie. He asked me how I enjoyed it and I told him that it 
 was not easy to like. My main objection was the lack of foundation for the relationship 
 between the two main characters, I was never convinced that they were close. I also 
 told him that the scene where the main characters were presented as children becoming
 friends was too late in the film.<br /><br />He told me that the flashback scenes 
 were not in the original script. That they were added because he felt like I did that
 the two main characters did not appear close. He went on to explain that these 
 scenes were not filmed to his satisfaction as they were out of money. I agree 
 that they did not do much for the film.<br /><br />Another fact about the movie,
 that I was not aware of, is the actor who had the lead wrote the script based 
 on his own personal experience. This is usually a bad move as some writers do 
 not take into consideration the emotional reaction the viewer. The story is so 
 close to home that the writer make too many assumption as to the audience's 
 reaction to his own tragedy. And the story is tragic. However, it did not work 
 for me as I never cared for any of the characters, least of all the lead. What 
 was presented were two evil people out to make a buck by any means, regardless 
 who gets hurt. When Ms. Young's character decides to give up he evil ways, it 
 appears that she does so because she is ineffective, not because she knows she 
 is doing wrong. If the movie has a message then I suspect that only the writer is aware of it.

The corresponding sentiment determined in the database is:  negative

The predicted sentiment for the randomly selected review is:  negative

Enter text for sentiment analysis (or press Enter to quit): Computer science is a field brimming with potential, offering high earning prospects, rapid career growth, and versatile skills. The demand for computer science professionals is consistently high across various industries, making it a lucrative and promising career path.
Predicted Sentiment: positive
________________________________