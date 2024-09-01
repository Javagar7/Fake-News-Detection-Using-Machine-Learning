Fake News Detection Using Machine Learning
Overview
This project is focused on building and evaluating machine learning models to classify news articles as either "fake" or "genuine" (real). Using datasets containing both fake and genuine news, the project preprocesses the text data, converts it into numerical representations using TF-IDF, and then applies machine learning algorithms like Logistic Regression and Passive Aggressive Classifier to classify the news articles.

Features
Data Preprocessing: Tokenization, stemming, and stopword removal to clean and prepare the text data.
TF-IDF Vectorization: Conversion of textual data into numerical features using Term Frequency-Inverse Document Frequency (TF-IDF).
Model Training: Implementation of Logistic Regression and Passive Aggressive Classifier models to predict the authenticity of news articles.
Accuracy Evaluation: Comparison of model performance based on accuracy scores.
Technologies Used
Python: Programming language for implementation.
NLTK: Natural Language Toolkit for text processing.
Scikit-learn: Machine learning library for model building and evaluation.
Pandas: Data manipulation and analysis.
TF-IDF Vectorizer: For feature extraction from text data.
Datasets
Fake.csv: Contains fake news articles.
True.csv: Contains genuine news articles.


Future Enhancements
Implement additional machine learning models (e.g., SVM, Random Forest).
Explore deep learning approaches using LSTM or CNNs.
Integrate with a web application for real-time news article classification.
