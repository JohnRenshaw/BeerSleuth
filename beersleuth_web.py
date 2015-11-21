import os
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





app = Flask(__name__)
app.secret_key = 'many random bytes'

# home page
@app.route('/', methods=['GET','POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html', title='BeerSleuth', beer_dict = json.dumps(beer_dict) , test= json.dumps('idle'))
    if request.method == 'POST':
        if request.form['beer button'] == 'Add' :

            return render_template('index.html', title='BeerSleuth', beer_dict = json.dumps(beer_dict), user_pred=0, beer_pred=0, pred=0)
        else:
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
            return render_template('index.html', title='BeerSleuth', beer_dict = json.dumps(beer_dict))





@app.route('/recs', methods=['POST'])
def get_recs():
    user = request.form['user_n2']
    beer = request.form['beer_p'].encode('utf-8')
    print(beer)
    print(users_df['user'].values)
    if user not in users_df['user'].values or beer not in beer_dict.keys():
        error_str =  "Unrecognized beer or user, have you added ratings and trained the model?"
        return render_template('index.html', title='BeerSleuth', beer_dict = json.dumps(beer_dict), error_str = error_str)
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

    return '''
            <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="utf-8">
            <meta http-equiv="X-UA-Compatible" content="IE=edge">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <meta name="description" content="">
            <meta name="author" content="">

            <title>BeerSleuth: Beer Recommender</title>

            <!-- Bootstrap Core CSS - Uses Bootswatch Flatly Theme: http://bootswatch.com/flatly/ -->
            <link href="../static/css/bootstrap.css" rel="stylesheet">

            <!-- Custom CSS -->
            <link href="../static/css/freelancer.css" rel="stylesheet">
            <link href="../static/css/style.css" rel="stylesheet">

            <!-- Custom Fonts -->
            <link href="../static/font-awesome/css/font-awesome.min.css" rel="stylesheet" type="text/css">
            <link href="http://fonts.googleapis.com/css?family=Montserrat:400,700" rel="stylesheet" type="text/css">
            <link href="http://fonts.googleapis.com/css?family=Lato:400,700,400italic,700italic" rel="stylesheet" type="text/css">

            <!-- HTML5 Shim and Respond.js IE8 support of HTML5 elements and media queries -->
            <!-- WARNING: Respond.js doesn't work if you view the page via file:// -->
            <!--[if lt IE 9]>
                <script src="https://oss.maxcdn.com/libs/html5shiv/3.7.0/html5shiv.js"></script>
                <script src="https://oss.maxcdn.com/libs/respond.js/1.4.2/respond.min.js"></script>
            <![endif]-->
        </head>
        <body>
                <br>
                 <div class="col-lg-8" .center-block>
                    %s
                    </div>
        </body>
            ''' %pred_df.to_html(classes = [".table"], index = False, float_format=lambda x:'%.1f' %x)


@app.route('/beerlist')
def beer_list():
    return json.dumps(beer_dict.keys())

@app.route('/running')
def running():
    return "Teaching computers to like beer, this could take up to 30 seconds, page wil refresh when done"

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
    else: jsonify(result = return_string)

if __name__ == '__main__':
    engine = create_engine('postgresql://postgres:123@localhost:5432/beersleuth')
    users_df = pd.read_sql_query('''SELECT DISTINCT mt3ratings.user, user_id FROM mt3ratings WHERE appdata = 1''', engine)
    cursor = engine.connect()
    q = cursor.execute('SELECT DISTINCT beer, beer_id FROM mt3beers')
    beer_dict = dict([(beer[0].encode('utf-8', "ignore"), int(beer[1])) for beer in q])
    rev_beer_dict = dict([(v, k) for k,v in beer_dict.iteritems()])
    #start spark
    conf = SparkConf() \
      .setAppName("BeerSleuthALS") \
      .set("spark.executor.memory", "4g")
    sc = SparkContext(conf=conf)
    sqlContext = SQLContext(sc)
    app.run(host='0.0.0.0', port=8080, debug=True)
