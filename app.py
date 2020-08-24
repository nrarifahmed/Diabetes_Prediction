# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 17:45:23 2020

@author: This-PC
"""
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('diabetes_classifier.pkl', 'rb'))

@app.route('/')
def home():
        return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='The Result is {} [ Note: 1=Diabetic 0=No-Diabetes]'.format(output))


if __name__ == "__main__":
        app.run(debug=True)
