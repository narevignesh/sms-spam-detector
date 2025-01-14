from flask import Flask, render_template, request, jsonify
from model import train_model, predict_new_data
from werkzeug.utils import quote


app = Flask(__name__)

# Train the model and get initial data
model, vectorizer, accuracy = train_model()

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None  # Variable to hold the prediction result
    if request.method == 'POST':
        sms_text = request.form.get('sms_text')
        if sms_text:
            prediction = predict_new_data(model, vectorizer, [sms_text])[0].item()  # Get prediction
        else:
            prediction = "Please enter some text for prediction."
    if prediction == 1:
        prediction = "Spam"
    elif prediction == 0:
        prediction = "Not Spam"
    else:
        prediction = "Please enter some text for prediction."
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
