import os
import tempfile
import psycopg2
import json
import pandas as pd
import shutil
import time
from flask import Flask, render_template, request, flash, redirect, url_for, jsonify
from sqlalchemy import create_engine, Table, MetaData
from sqlalchemy.exc import DataError, IntegrityError
from pyspark import SparkConf, SparkContext
from pyspark import SQLContext, HiveContext
from pyspark.mllib.recommendation import MatrixFactorizationModel
import beer_spark as modeling

engine = create_engine('postgresql://postgres:123@localhost:5432/beersleuth')
cursor = engine.connect()
q = cursor.execute('SELECT DISTINCT beer, beer_id FROM mt3beers')
beer_dict = dict([(beer[0].encode('utf-8', "ignore"), int(beer[1])) for beer in q])
rev_beer_dict = dict([(v, k) for k,v in beer_dict.iteritems()])
    #start spark
os.environ["SPARK_HOME"] = "/home/ubuntu/spark-1.5.2-bin-hadoop2.6"
conf = SparkConf().setAppName("BeerSleuthALS").set("spark.executor.memory", "4g")
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)
path = tempfile.mkdtemp()



app = Flask(__name__)
app.secret_key = 'many random bytes'

# home page
@app.route('/')
def index():
    return render_template('index.html', title='BeerSleuth')

@app.route('/get_ratings')
def get_ratings():
    usern = request.args.get('usern')
    try: key = request.args.get('key')
    except NameError: key = 'e'
    if key == 'abcd':
        query = "SELECT beer, taste FROM mt3ratings WHERE mt3ratings.user = '%s'" %usern
        user_ratings = pd.read_sql_query(query, engine)
        return  jsonify(result = user_ratings.to_html(index=False, col_space = "50%", classes = "table-hover"))


@app.route('/fitting')
def ALS_fit():
    try: key = request.args.get('key')
    except NameError: key = 'e'
    if key == 'abcd':
        ratings_sqldf = modeling.get_item_user_rev_from_pg(engine, sqlContext)
        sqlContext.registerDataFrameAsTable(ratings_sqldf, "ratings")
        print('fitting model')
	model = modeling.fit_final_model(ratings_sqldf)
	print('save model')
        model.save(sc, path)
	print('done')
        return jsonify(result="Model training complete, you may now get predictions")


@app.route('/beerlist')
def beer_list():
    return json.dumps(beer_dict.keys())

@app.route('/running')
def running():
    return "Teaching computers to like beer, this could take up to 30 seconds, page will refresh when done"

@app.route('/userlist')
def user_list():
    return json.dumps(users_df['user'].values.tolist())

@app.route('/ratings_form_data')
def add_rating():
    i=0
    return_string = "No ratings added"
    usern = request.args.get('usern')
    beer1 = request.args.get('beer1')
    br1 = request.args.get('br1')
    beer2 = request.args.get('beer2')
    br2 = request.args.get('br2')
    beer3 = request.args.get('beer3')
    br3 = request.args.get('br3')
    try: key = request.args.get('key')
    except NameError: key = 'e'
    if key == 'abcd':
        if br1 and beer1:
            try:
                beer_id1=  beer_dict[beer1.encode('utf-8')]
                modeling.add_rating_to_db(usern, beer1, beer_id1, br1, engine)
                i += 1
            except  (KeyError, DataError, IntegrityError): pass
        if br2 and beer2:
            try:
                beer_id2=  beer_dict[beer2.encode('utf-8')]
                modeling.add_rating_to_db(usern, beer2, beer_id2, br2, engine)
                i += 1
            except  (KeyError, DataError, IntegrityError): pass
        if br3 and beer3:
            try:
                beer_id3=  beer_dict[beer3.encode('utf-8')]
                modeling.add_rating_to_db(usern, beer3, beer_id3, br3, engine)
                i += 1
            except (KeyError, DataError, IntegrityError): pass
        return_string = '%i ratings added to db'%i
        return jsonify(result = return_string)
    else: return jsonify(result = return_string)

@app.route('/prediction')
def prediction():
    i=0
    user_p = request.args.get('user_p')
    beer_p = request.args.get('beer_p').encode('utf-8')
    try: key = request.args.get('key')
    except NameError: key = 'e'
    if key == 'abcd':
            users_df = pd.read_sql_query('''SELECT DISTINCT mt3ratings.user, user_id FROM mt3ratings WHERE appdata = 1''', engine)
            if user_p not in users_df['user'].values:
                return_str =  "can't find user"
                return jsonify(result = return_str)

            if beer_p not in beer_dict.keys():
		return_str = "can't find beer"
		return jsonify(result = return_str)
            user_id = users_df.user_id[users_df.user == user_p].values[0]
            beer_id = beer_dict[beer_p]
            print user_p, beer_p, user_id, beer_id
            model = MatrixFactorizationModel.load(sc, path)
            pred = model.predict(user_id, beer_id)
            return_str = "Prediction: %0.1f"%pred
            return jsonify(result = return_str)

@app.route('/top20')
def prediction():
    i=0
    user_p = request.args.get('user_p')
    try: key = request.args.get('key')
    except NameError: key = 'e'
    if key == 'abcd':
        user_id = users_df.user_id[users_df.user == user_p].values[0]
        model = MatrixFactorizationModel.load(sc, path)
        pred_df = pd.DataFrame(model.recommendProducts(user_id, 20), columns = ['user', 'beer_id', 'rating'])
        return  jsonify(result = pred_df.to_html(index=False, col_space = "50%", classes = "table-hover"))


if __name__ == '__main__':
   app.run(host='0.0.0.0',  debug=False)
