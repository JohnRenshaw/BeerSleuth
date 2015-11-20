import os
import psycopg2
import json
import pandas as pd
import shutil
from flask import Flask, render_template, request
from sqlalchemy import create_engine, Table, MetaData
from pyspark import SparkConf, SparkContext
from pyspark import SQLContext, HiveContext
from pyspark.mllib.recommendation import MatrixFactorizationModel
import beer_spark as modeling





app = Flask(__name__)


# home page
@app.route('/')
def index():
    return render_template('index.html', title='Hello!', beer_list = json.dumps(list(beer_dict.keys())), user_pred=0, beer_pred=0, pred=0 )

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
    modeling.add_rating_to_db(username, beer_id1, br1, engine)
    modeling.add_rating_to_db(username, beer_id2, br2, engine)
    modeling.add_rating_to_db(username, beer_id3, br3, engine)
    return render_template('index.html', title='Hello!', beer_dict = json.dumps(beer_dict), user_pred=0, beer_pred=0, pred=0)

@app.route('/predict', methods=['POST'])
def get_preds():
    ratings_sqldf = modeling.get_item_user_rev_from_pg(engine, sqlContext)
    sqlContext.registerDataFrameAsTable(ratings_sqldf, "ratings")
    model = modeling.fit_final_model(ratings_sqldf)
    path = os.getcwd()
    shutil.rmtree('metadata', ignore_errors=True)
    shutil.rmtree('data', ignore_errors=True)
    model.save(sc, path)
#    beer_id_list = beer_dict.values()
#    u_id_df = pd.read_sql_query('''SELECT DISTINCT slist.user_id FROM (SELECT mt3ratings.user, user_id FROM mt3ratings WHERE appdata = 1) slist''', engine)
#    predict_list = [(u_id, b_id ) for b_id in beer_id_list for u_id in u_id_df.user_id.values]
#    pred_rdd = sc.parallelize(predict_list)
#    pred_list = model.predictAll(pred_rdd).collect()
#    for pred in pred_list:
#        modeling.add_pred_to_db(pred[0], pred[1], pred[2], engine)
    return "model has been fit" 

@app.route('/recs', methods=['POST'])
def get_recs():
    users_df = pd.read_sql_query('''SELECT DISTINCT mt3ratings.user, user_id FROM mt3ratings WHERE appdata = 1''', engine)
    user = request.form['user_n2']
    beer = request.form['beer_p']
    user_id = users_df.user_id[users_df.user == user].values[0]
    beer_id = beer_dict[beer]
    path = os.getcwd()
    model = MatrixFactorizationModel.load(sc, path)
    pred = model.predict(user_id, beer_id)
    pred_df = pd.DataFrame({
        'user':[user],
        'beer':[beer],
        'prediction':[pred]
        }, columns=['user', 'beer', 'prediction'])

    return pred_df.to_html()


@app.route('/beerlist')
def beer_list():
    return json.dumps(beer_dict.keys())



if __name__ == '__main__':
    engine = create_engine('postgresql://postgres:123@localhost:5432/beersleuth')
    cursor = engine.connect()
    q = cursor.execute('SELECT DISTINCT beer, beer_id FROM mt3beers')
    beer_dict = dict([(beer[0].encode('utf-8'), int(beer[1])) for beer in q])
    rev_beer_dict = dict([(v, k) for k,v in beer_dict.iteritems()])
    #start spark
    conf = SparkConf() \
      .setAppName("BeerSleuthALS") \
      .set("spark.executor.memory", "4g")
    sc = SparkContext(conf=conf)
    sqlContext = SQLContext(sc) 
    app.run(host='0.0.0.0', port=8080, debug=True)

