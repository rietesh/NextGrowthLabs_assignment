import os
from flask import Flask, flash, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from happytransformer import HappyTextToText

happy_tt = HappyTextToText("T5", "vennify/t5-base-grammar-correction")
UPLOAD_FOLDER = "./uploads"
ALLOWED_EXTENSIONS = {'csv'}
df = pd.read_csv('./grammar_results.csv')
df = df[df['is_Error']==1]

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_results(path):
    data = pd.read_csv(path)
    text = data['Text'].astype(str).values.tolist()
    sentiments = SentimentIntensityAnalyzer()
    vader = []
    blob = []
    for i in text:
        sentiment_dict = sentiments.polarity_scores(i)
        vader.append(sentiment_dict)
        blob.append(TextBlob(i).sentiment.polarity)

    results = pd.DataFrame(vader)
    results['text'] = text
    results['star'] = data['Star'].values
    results['blob'] = blob
    results['sent_res'] = results.apply(lambda x: ['neg','neu','pos'][np.argmax([x['neg'],x['neu'], x['pos']])], axis=1)
    results['star_res'] = results['star'].apply(lambda x: 'pos' if x>3 else ('neu' if x==3 else 'neg'))
    pos_neg = results[(results['sent_res']=='pos') & (results['star_res']=='neg')][['text','star','sent_res']].reset_index()
    neg_pos = results[(results['sent_res']=='neg') & (results['star_res']=='pos')][['text','star','sent_res']].reset_index()
    answer = pd.concat([neg_pos,pos_neg])
    return answer

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            answer = get_results(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return render_template("table.html", tables=[answer.reset_index().to_html()], titles=[''])

    return '''
    <!doctype html>
    <title>Sentiment Mismatch</title>
    <h1>Sentiment Mismatch Application</h1>
    <h2>Upload csv File</h2>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

def get_correct_sent(text):
    text = "grammar: "+text
    result = happy_tt.generate_text(text)
    return result.text


@app.route('/grammar_check/', methods=['GET', 'POST'])
def check_grammar():
    if request.method == 'POST':
        res = get_correct_sent(request.form['fname'])
        d = pd.DataFrame({'input': [request.form['fname']], 'Grammar Corrected': [res]})
        return render_template("table.html", tables=[d.to_html()], titles=[''])

    if request.method == 'GET':
        return render_template("table.html", tables=[df.to_html()], titles=[''])

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 80)))
