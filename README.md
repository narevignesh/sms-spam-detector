Here's a GitHub `README.md` description for the machine learning model that powers your SMS Spam Detector:

```markdown
# SMS Spam Detection Machine Learning Model

This project leverages machine learning to classify SMS messages as "Spam" or "Not Spam" (ham) using a Naive Bayes classifier. The model is trained on the **SMSSpamCollection** dataset, which contains labeled examples of spam and non-spam SMS messages.

## Model Overview

- **Model Type**: Naive Bayes (MultinomialNB)
- **Vectorization**: Text data is vectorized using **CountVectorizer**, which converts the text into a bag-of-words representation.
- **Evaluation Metric**: The model is evaluated using **accuracy**.

## Features

- **SMS Text Classification**: The model classifies SMS messages as spam or not spam based on the content of the message.
- **Training Data**: The model is trained on a dataset containing labeled SMS messages.
- **Web Interface**: Flask web application allows users to enter SMS text and get real-time predictions.

## Installation

### Requirements

- Python 3.x
- `flask`
- `scikit-learn`
- `pandas`

You can install the required dependencies by running:

```bash
pip install flask scikit-learn pandas
```

### Setup

1. Clone the repository:

    ```bash
    git clone https://github.com/narevignesh/sms-spam-detector.git
    cd sms-spam-detector
    ```

2. Download the **SMSSpamCollection** dataset and place it in the project directory. You can find the dataset [here](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection).

3. Run the application:

    ```bash
    python model.py
    python app.py
    ```

4. Access the web app at `http://127.0.0.1:5000/`.

## How the Model Works

### 1. Dataset

The model is trained using the **SMSSpamCollection** dataset, which contains labeled SMS messages. Each message is labeled as either `ham` (non-spam) or `spam`. The dataset is tab-separated and consists of two columns: one for the label and one for the message content.

### 2. Data Preprocessing

- The **CountVectorizer** is used to convert the SMS text into a matrix of token counts. It converts each SMS message into a "bag-of-words" representation.


### 3. Model Training

- The dataset is split into training and test sets using `train_test_split` from `scikit-learn`.
- A **Multinomial Naive Bayes** classifier (`MultinomialNB`) is trained on the vectorized training data.
- The accuracy of the model is evaluated using the test set and `accuracy_score` from `scikit-learn`.

### 4. Model Evaluation

- The accuracy of the model on the test data is calculated and returned as a percentage, providing an indicator of how well the model performs.

### 5. Predicting SMS Messages

Once the model is trained, it can predict whether new SMS messages are spam or not. The app takes SMS input, transforms it using the same vectorizer, and predicts the label (spam or not spam).

## Example Usage

### Example Prediction:

#### Input:
```
"Congratulations! You've won a $1000 prize."
```

#### Output:
```
Prediction: Spam
```

#### Input:
```
"Hey, are we still meeting tomorrow?"
```

#### Output:
```
Prediction: Not Spam
```

## Acknowledgments

- The **SMSSpamCollection** dataset is used for training the model.
- The app is built using **Flask** for the web interface, and **scikit-learn** for machine learning.
```

### Key Sections:

- **Model Overview**: Describes the machine learning model and the tools used (Naive Bayes, CountVectorizer).
- **Features**: Summarizes the functionality of the app.
- **Installation**: Provides steps to install dependencies and set up the project.
- **How the Model Works**: Details the model training process, including data preprocessing, training, and evaluation.
- **Example Usage**: Provides code examples for training the model and making predictions.
- **License and Acknowledgments**: Credits and licensing information.

This README gives a clear and structured explanation of your SMS Spam Detection machine learning model, making it easy for others to understand and use the project.
