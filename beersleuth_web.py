from flask import Flask, render_template, request
from sqlalchemy import create_engine, Table, MetaData
import psycopg2
import json
import pandas as pd
import beer_spark as modeling
from pyspark import SparkConf, SparkContext
from pyspark import SQLContext, HiveContext




app = Flask(__name__)


# home page
@app.route('/')
def index():
    return render_template('index.html', title='Hello!', beer_list = json.dumps(list(beer_dict.keys())) )

@app.route('/', methods=['POST'])
def get_ratings_1():
    username = request.form['usern'].encode('utf-8')
    beer1 = request.form['beer1']
    beer2 = request.form['beer2']
    beer3 = request.form['beer3']
    br1 = request.form['br1']
    br2 = request.form['br2']
    br3 = request.form['br3']
    beer_id1=  beer_dict[beer1.encode('utf-8')]
    beer_id2=  beer_dict[beer2.encode('utf-8')]
    beer_id3=  beer_dict[beer3.encode('utf-8')]
    modeling.add_rating_to_db(username, beer_id1, br1, engine, users_df, beer_df)
    modeling.add_rating_to_db(username, beer_id2, br2, engine, users_df, beer_df)
    modeling.add_rating_to_db(username, beer_id3, br3, engine, users_df, beer_df)
#    ratings_sqldf = modeling.get_item_user_rev_from_pg(engine, sqlContext)
#    sqlContext.registerDataFrameAsTable(ratings_sqldf, "ratings")
#    model = modeling.fit_final_model(ratings_sqldf)
    return render_template('index.html', title='Hello!', beer_dict = json.dumps(beer_dict))

@app.route('/beerlist')
def beer_list():
    return json.dumps(beer_dict.keys())



if __name__ == '__main__':
    engine = create_engine('postgresql://postgres:123@localhost:5432/beersleuth')
    cursor = engine.connect()
    q = cursor.execute('SELECT DISTINCT beer, beer_id FROM mt3beers')
    beer_dict = dict([(beer[0].encode('utf-8'), int(beer[1])) for beer in q])
    users_df, beer_df = modeling.get_app_user_beer_id_pairs(engine)
    app.run(host='0.0.0.0', port=8080, debug=True)
    
    
#    
#    #start spark
#    conf = SparkConf() \
#      .setAppName("BeerSleuthALS") \
#      .set("spark.executor.memory", "4g")
#    sc = SparkContext(conf=conf)
#    sqlContext = SQLContext(sc) 

