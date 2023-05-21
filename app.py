import pickle
from flask import Flask, render_template, request, redirect, url_for, flash
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from string import punctuation
import re
import numpy as np
from nltk.corpus import stopwords

nltk.download('stopwords')

set(stopwords.words('english'))

# https://app.surgehq.ai/datasets/toxicity?taskResponseId=
app = Flask(__name__)
app.secret_key = 'secret_key'  

# Load the trained toxicity analysis model
toxicity_model = pickle.load(open('toxicity_analysis.pkl', 'rb'))
# Load the trained toxicity analysis model
vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))

@app.route('/')
def my_form():
    return render_template('login.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == 'admin' and password == 'pass':
            return redirect(url_for('form'))
        else:
            flash('Invalid username or password')
            return redirect(url_for('login'))
    return render_template('login.html')

@app.route('/form')
def form():
    return render_template('form.html')

@app.route('/sentiment', methods=['POST'])
def sentiment():
    stop_words = stopwords.words('english')
    
    #convert to lowercase
    text1 = request.form['text1'].lower()
    
    text_final = ''.join(c for c in text1 if not c.isdigit())
          
    #remove stopwords    
    processed_doc1 = ' '.join([word for word in text_final.split() if word not in stop_words])

    sa = SentimentIntensityAnalyzer()
    dd = sa.polarity_scores(text=processed_doc1)
    compound = round((1 + dd['compound'])/2, 2)

     # Convert the processed text into numerical features using the same vectorizer
    processed_text_features = vectorizer.transform([processed_doc1])

    # Classify the text using the toxicity analysis model
    prediction = toxicity_model.predict(processed_text_features)

    print(prediction)

    return render_template('form.html', final=compound, text1=text_final, text2=dd['pos'], text5=dd['neg'], text4=compound, text3=dd['neu'], prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5002, threaded=True)
