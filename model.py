from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import pandas as pd

def train_model():
    # Load dataset
    data = pd.read_csv('SMSSpamCollection', sep='\t', names=['label', 'message'])
    data['label'] = data['label'].map({'ham': 0, 'spam': 1})
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        data['message'], data['label'], test_size=0.2, random_state=42
    )
    
    # Vectorize text
    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Train model
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, vectorizer, accuracy

def predict_new_data(model, vectorizer, messages):
    messages_vec = vectorizer.transform(messages)
    return model.predict(messages_vec)
