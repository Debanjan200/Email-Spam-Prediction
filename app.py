from flask import Flask,render_template,request
from nltk.stem.porter import PorterStemmer
import pickle,re

log_classifier=pickle.load(open("log_classifier.pkl","rb"))
cv=pickle.load(open("vectorizer.pkl","rb"))
with open("stopwords.txt","r") as f:
    stopwords=f.read().split("\n")

app=Flask(__name__)

def preprocessing(str):
    corpus=[]
    ps=PorterStemmer()
    emo=re.sub("[^a-zA-Z]",' ',str)
    emo=emo.lower()
    emo=emo.split()
    emo=[ps.stem(word) for word in emo if word not in set(stopwords)]
    emo=' '.join(emo)
    corpus.append(emo)
    data=cv.transform(corpus)
    return prediction(data)[0]

def prediction(lst):
    return log_classifier.predict(lst)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/",methods=["POST"])
def predict():
    if request.method=="POST":
        email=request.form["email"]

        pred=preprocessing(email)

    if pred==0:
        msg="The mail is ham"

    else:
        msg="The mail is spam"

    return render_template("index.html",prediction=pred,st=msg)


if __name__=="__main__":
    app.run(debug=True)