from flask import render_template, request
from app import app

import pickle
import pandas as pd

import tensorflow
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences

import keras
from keras import models
from keras.models import load_model


@app.route("/", methods = ["GET", "POST"])
def index():
    msg = None
    if(request.method == "POST"):
        if request.method == 'POST':
            text = request.form['text']
            print(text)
            model = load_model('percobaan4.h5')
            tokenizer = pickle.load(open("tokenizer", "rb"))
            news_classes = ['Clickbait', 'False', 'True']
            max_len=149
            testing_news = {"text":[text]}
            new_def_test = pd.DataFrame(testing_news)
            new_x_test = new_def_test["text"]
            print(new_x_test)

            xt = tokenizer.texts_to_sequences([text])
            xt = sequence.pad_sequences(xt, padding='post', maxlen=max_len)
            print(xt)

            yt = model.predict(xt).argmax(axis=1)
            print(yt)
            print('The predicted news is', news_classes[yt[0]])
            
            if yt[0] == 0:
                result = "A Clickbait News"
            elif yt[0] == 1:
                result = "A Fake News"
            elif yt[0] == 2:
                result = "Not A Fake News"

            return render_template("hasil.html", msg = result)
        else:
            msg = "Username is not available"

    return render_template("index.html", msg = msg)
