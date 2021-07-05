import numpy as np
import logging
from flask import Flask, request, jsonify, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

app = Flask(__name__)
model = pickle.load(open('Naive_Bayes_Classifier.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST', 'GET'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
   
    vectorizer = TfidfVectorizer()
    tfidf_test_ = vectorizer.transform(input(''))
    prediction = model.predict(tfidf_test_)

    #output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Predicted Name $ {}'.format(prediction))


if __name__ == "__main__":
    app.run(debug=True)