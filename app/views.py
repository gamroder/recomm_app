from app import app
from app.recomm import recomm_engine
from flask import Flask,render_template,request
import sys
reload(sys)
sys.setdefaultencoding("utf-8")
@app.route('/',methods = ["GET","POST"])
def main():
    predicted = {}
    if request.method == 'POST':
        #if request.form['url']!='':
        n_film = request.form['url']
        #request.form['text']
        test_d=recomm_engine()
        predicted=test_d.predict(n_film,5)
        return render_template('begin.html', results = predicted)
    #if request.method == 'GET':