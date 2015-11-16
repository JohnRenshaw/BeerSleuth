
import sys
import itertools
import math
from operator import add
from os.path import join, isfile, dirname
from pyspark import SparkConf, SparkContext
from pyspark import SQLContext, HiveContext
from pyspark.mllib.recommendation import MatrixFactorizationModel, ALS
from pyspark.sql.types import *
import pandas as pd
import numpy as np
from pymongo import MongoClient
from sqlalchemy import create_engine, Table, MetaData
import psycopg2

client = MongoClient()
db = client['ratebeer']
beer_ratings = db.ratings

def parseRating(ratings_file):
    """
    parses a beer ratings of the format taste, user_id, beer_id
    """
    ratings_raw_data = sc.textFile(ratings_file)
    ratings_raw_data_header = ratings_raw_data.take(1)[0]
    ratings_data = ratings_raw_data.filter(lambda line: line != ratings_raw_data_header)\
        .map(lambda line: line.split(",")).map(lambda tokens: (tokens[0], tokens[1], tokens[2])).cache()
    return ratings_data


def mongo_to_csv():
    #create users csv
    users = beer_ratings.distinct('user')
    users_df= pd.DataFrame({'user_id':range(len(users)), 'user':users})
    users_df.user =users_df.user.apply(lambda x:str(x.encode('UTF-8','ignore')))
    users_df.to_csv('users_spark.csv',index=False )

    #create beers csv
    beers = beer_ratings.distinct('beer')
    beers_df= pd.DataFrame({'beer_id':range(len(beers)), 'beer':beers})
    beers_df.beer = beers_df.beer.apply(lambda x:str(x.encode('UTF-8','ignore')))
    beers_df.to_csv('beers_spark.csv',index=False )
    #create ratings csv

    ratings_df = pd.DataFrame(list(beer_ratings.find({},{'user':1,'beer':1,'taste':1})))
    ratings_df.beer = ratings_df.beer.apply(lambda x:str(x.encode('UTF-8','ignore')))
    ratings_df.user = ratings_df.user.apply(lambda x:str(x.encode('UTF-8','ignore')))
    ratings_df = ratings_df[ratings_df['taste']!='NA'].merge(users_df, how='left').merge(beers_df, how='left').drop(['_id', 'beer', 'user'], axis=1)
    ratings_df = ratings_df[['user_id', 'beer_id', 'taste']]
    ratings_df['taste'] = ratings_df['taste']*10
    ratings_df.to_csv('ratings_spark.csv', index=False, encoding='UTF-8' )

def mongo_to_df():
    #create users csv
    users = beer_ratings.distinct('user')
    users_df= pd.DataFrame({'user_id':range(len(users)), 'user':users})


    #create beers csv
    beers = beer_ratings.distinct('beer')
    beers_df= pd.DataFrame({'beer_id':range(len(beers)), 'beer':beers})

    #create ratings csv
    ratings_df = pd.DataFrame(list(beer_ratings.find({},{'user':1,'beer':1,'taste':1})))
    ratings_df = ratings_df[ratings_df['taste']!='NA'].merge(users_df, how='left').merge(beers_df, how='left').drop(['_id', 'beer', 'user'], axis=1)
    ratings_df = ratings_df[['user_id', 'beer_id', 'taste']]
    ratings_df['taste'] = ratings_df['taste']*10
    return ratings_df

def get_item_user_rev_from_pg(engine):
    taste_df = pd.read_sql_query('''
        SELECT * FROM ratings 
        WHERE ratings.user NOT IN 
            (SELECT counts.reviewer FROM  
                (SELECT ratings.user as reviewer,count(*) FROM ratings GROUP BY ratings.user)
                 as counts WHERE counts.count < 4)
         ''', engine)

    taste_df = taste_df[['user_id', 'beer_id', 'taste']]
    taste_df.taste = taste_df.taste.astype(int)
    return sqlContext.createDataFrame(taste_df)


def model_param_sweep(train, test):
    #model params
    iterations = 10
    regularization_param_list = np.linspace(0.05, 0.2, 5)

    #params used in keeping track of error between different ranks
    rank_list = [4, 6, 8]
    errors = np.zeros(len(regularization_param_list)*len(rank_list))
    err = 0
    min_error = float('inf')
    best_rank = -1
    best_iteration = -1

    for rank in rank_list:
        for reg in regularization_param_list:
            model = ALS.train(train.rdd.map(lambda x: (x[0], x[1], x[2])), rank=rank, nonnegative=True, iterations=iterations, lambda_=reg)
            predictions =  model.predictAll(test.rdd.map(lambda r: (r[0], r[1]) )).map(lambda x: ((int(x[0]), int(x[1])), float(x[2])) )
            rates_and_preds = test.rdd.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
            error = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean())
            errors[err] = error
            err += 1
            print 'For rank=%s, regParam=%s the RMSE is %s' % (rank, reg, error)
            if error < min_error:
                min_error = error
                best_rank = (rank, reg)
    print 'The best model was trained with (rank, regParam): %s' %str(best_rank)

def fit_final_model(train):
    #model params
    iterations = 15
    reg = 0.175
    rank = 4
    model = ALS.train(train.rdd.map(lambda x: (x[0], x[1], x[2])), rank=rank, nonnegative=False, iterations=iterations, lambda_=reg)
    predictions =  model.predictAll(test.rdd.map(lambda r: (r[0], r[1]) )).map(lambda x: ((int(x[0]), int(x[1])), float(x[2])) )


def get_user_beer_id_pairs(engine):
    users_df = pd.read_sql_query('''SELECT DISTINCT ratings.user, user_id FROM ratings''', engine)
    beer_df = pd.read_sql_query('''SELECT DISTINCT beer, beer_id FROM ratings''', engine)
    return users_df, beer_df

def add_rating_to_db(user, beer, taste, engine):
    users_df, beer_df = get_user_beer_id_pairs(engine)
    metadata = MetaData(engine)
    ratings = Table('ratings', metadata, autoload=True)
    if user not in users_df.user.values:
        num_users = pd.read_sql_query('''SELECT DISTINCT count(ratings.user) FROM ratings''', engine)
        user_id = num_users['count'].values[0]
    else: user_id = users_df.user_id[users_df.user == user].values[0]
    beer_id = beer_df.beer_id[beer_df.beer == beer].values[0]
    _id = beer + '_' + user
    i = ratings.insert()
    i.execute(_id=_id,beer=beer,taste=taste,user=user, user_id=user_id, beer_id=beer_id )
    print 'rating added successfully'


if __name__ == '__main__':
    # set up environment
    conf = SparkConf() \
      .setAppName("BeerSleuthALS") \
      .set("spark.executor.memory", "4g")
    sc = SparkContext(conf=conf)
    sqlContext = SQLContext(sc) 

    #load data
    engine = create_engine('postgresql://postgres:123@localhost:5432/beersleuth')
    ratings_sqldf = get_item_user_rev_from_pg(engine)
    sqlContext.registerDataFrameAsTable(ratings_sqldf, "ratings")
    train, test = sqlContext.table('ratings').randomSplit([.8, .2])
    train = train.cache()
    test = test.cache()
#    add_rating_to_db(user='johnjohn', beer=u'101 North Heroine IPA' , taste=8, engine=engine)
#    add_rating_to_db(user='johnjohn', beer=u'Boulder Creek Golden Promise' , taste=6, engine=engine)
    model_param_sweep(train, test)
#    import timeit
#    start_time = timeit.default_timer()
#    fit_final_model(ratings_sqldf)
#    elapsed = timeit.default_timer() - start_time
'''
initial CA ratings db had 627431 ratings
'''