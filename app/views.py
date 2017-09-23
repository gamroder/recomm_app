from app import app
from app.recomm import recomm_engine
from flask import Flask,render_template
import sys
reload(sys)
sys.setdefaultencoding("utf-8")
@app.route('/')
@app.route('/index')
def main():
	test_d=recomm_engine()
	predicted=test_d.predict(1,5)
	return render_template('index.html', result = predicted)

