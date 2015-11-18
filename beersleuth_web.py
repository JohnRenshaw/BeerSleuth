from flask import Flask, render_template, request
from sqlalchemy import create_engine, Table, MetaData
from flask_wtf import Form
from wtforms import StringField
from wtforms.validators import DataRequired
import psycopg2
import json
import pandas as pd

engine = create_engine('postgresql://postgres:123@localhost:5432/beersleuth')
#beer_df = pd.read_sql_query('SELECT DISTINCT beer FROM mt3ratings', engine)
#beer_list = beer_df.values.ravel().tolist()

app = Flask(__name__)
cursor = engine.connect()
q = cursor.execute('SELECT DISTINCT beer FROM mt3ratings')
beer_list = [beer[0].encode('utf-8') for beer in q]

# home page
@app.route('/')
def index():
    return render_template('index.html', title='Hello!', beer_list = json.dumps(beer_list) )

@app.route('/', methods=['POST'])
def get_ratings_1():
    beer1 = request.form['beer1']
    beer2 = request.form['beer2']
    beer3 = request.form['beer3']
    br1 = request.form['br1']
    br2 = request.form['br2']
    br3 = request.form['br3']
    print beer1, beer2, beer3
    return render_template('index.html', title='Hello!', beer_list = json.dumps(beer_list))

#@app.route('/more/')
#def more():
#    return render_template('starter_template.html')#


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
