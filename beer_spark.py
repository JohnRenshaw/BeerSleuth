
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
from sqlalchemy import create_engine, Table, MetaData
import psycopg2


def parseRating(ratings_file):
    """
    parses a beer ratings of the format taste, user_id, beer_id
    """
    ratings_raw_data = sc.textFile(ratings_file)
    ratings_raw_data_header = ratings_raw_data.take(1)[0]
    ratings_data = ratings_raw_data.filter(lambda line: line != ratings_raw_data_header)\
        .map(lambda line: line.split(",")).map(lambda tokens: (tokens[0], tokens[1], tokens[2])).cache()
    return ratings_data


def get_item_user_rev_from_pg(engine, sqlContext):
    taste_df = pd.read_sql_query('''
        SELECT user_id, beer_id, taste FROM mt3ratings 
         ''', engine)
    taste_df.taste = taste_df.taste.astype(int)
    return sqlContext.createDataFrame(taste_df)


def model_param_sweep(train, test):
    #model params
    iterations = 20
    regularization_param_list = np.linspace(0.05, 0.2, 5)

    #params used in keeping track of error between different ranks
    rank_list = [4, 6, 8]
    errors = np.zeros(len(regularization_param_list)*len(rank_list))
    err = 0
    min_error = float('inf')
    max_class_rate = 0
    best_rank = -1
    best_iteration = -1

    for rank in rank_list:
        for reg in regularization_param_list:
            model = ALS.train(train.rdd.map(lambda x: (x[0], x[1], x[2])), rank=rank, nonnegative=True, iterations=iterations, lambda_=reg)
            predictions =  model.predictAll(test.rdd.map(lambda r: (r[0], r[1]) )).map(lambda x: ((int(x[0]), int(x[1])), float(x[2])) )
            rates_and_preds = test.rdd.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
            correct_count = rates_and_preds.filter(lambda r:( abs(r[1][0] - r[1][1]) < 1) or (r[1][0] < 6 and r[1][1] < 6) ).count()
            total_count = rates_and_preds.count()
            class_rate = correct_count*1./total_count
            error = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean())
            errors[err] = error
            err += 1
            print 'For rank=%s, regParam=%s the RMSE is %s with a correct classification rate of %0.3f' % (rank, reg, error, class_rate)
            if class_rate > max_class_rate:
                max_class_rate = class_rate
                best_rank = (rank, reg)
    print 'The best model was trained with (rank, regParam): %s and had class rate %0.3f' %(str(best_rank), max_class_rate)

def fit_final_model(train):
    #model params
    iterations = 20
    reg = 0.0875
    rank = 6
    model = ALS.train(train.rdd.map(lambda x: (x[0], x[1], x[2])), rank=rank, nonnegative=True, iterations=iterations, lambda_=reg)
    return model

def get_user_beer_id_pairs(engine):
    users_df = pd.read_sql_query('''SELECT DISTINCT mt3ratings.user, user_id FROM mt3ratings''', engine)
  #  beer_df = pd.read_sql_query('''SELECT DISTINCT mt3ratings.beer, mt3ratings.beer_id, abv, calories, count, style, brewery FROM mt3ratings INNER JOIN beers ON mt3ratings.beer_id = beers.beer_id INNER JOIN beercounts on beercounts.beer_id = beers.beer_id;''', engine)
    beer_df = pd.read_sql_query('''SELECT DISTINCT beer, beer_id FROM mt3beers''', engine)
    return users_df, beer_df

def get_app_user_beer_id_pairs(engine):
    users_df = pd.read_sql_query('''SELECT DISTINCT mt3ratings.user, user_id FROM mt3ratings WHERE appdata = 1''', engine)
  #  beer_df = pd.read_sql_query('''SELECT DISTINCT mt3ratings.beer, mt3ratings.beer_id, abv, calories, count, style, brewery FROM mt3ratings INNER JOIN beers ON mt3ratings.beer_id = beers.beer_id INNER JOIN beercounts on beercounts.beer_id = beers.beer_id;''', engine)
    beer_df = pd.read_sql_query('''SELECT DISTINCT beer, beer_id FROM mt3beers''', engine)
    return users_df, beer_df


def get_latent_beers(model, engine):
    latents = pd.DataFrame(model.productFeatures().map(lambda row: [row[1][0], row[1][1], row[1][2], row[1][3], row[1][4], row[1][5]]).collect())
    users_df, beer_df = get_user_beer_id_pairs(engine)
    l1 = beer_df.ix[np.argsort(latents[0])[::-1]]
    l2 = beer_df.ix[np.argsort(latents[1])[::-1]]
    l3 = beer_df.ix[np.argsort(latents[2])[::-1]]
    l4 = beer_df.ix[np.argsort(latents[3])[::-1]]
    l5 = beer_df.ix[np.argsort(latents[4])[::-1]]
    l6 = beer_df.ix[np.argsort(latents[5])[::-1]]
    l1['l1_rank']=range(1,len(l1)+1)
    l2['l2_rank']=range(1,len(l2)+1)
    l3['l3_rank']=range(1,len(l3)+1)
    l4['l4_rank']=range(1,len(l4)+1)
    l5['l5_rank']=range(1,len(l5)+1)
    l6['l6_rank']=range(1,len(l6)+1)
    combined = l1.merge(l2).merge(l3).merge(l4).merge(l5).merge(l6)
    return combined


def add_rating_to_db(user, beer, beer_id, taste, engine,preds=0):
    users_df = pd.read_sql_query('''SELECT DISTINCT mt3ratings.user, user_id FROM mt3ratings WHERE appdata = 1''', engine)
    metadata = MetaData(engine)
    ratings = Table('mt3ratings', metadata, autoload=True)
    if user not in users_df.user.values:
        num_users = pd.read_sql_query('''SELECT max(user_id) as users FROM mt3ratings''', engine)
        user_id = num_users['users'].values[0]+1000000
    else: user_id = users_df.user_id[users_df.user == user].values[0]
    _id = beer + '_' + user
    i = ratings.insert()
    try: i.execute(_id=_id,beer=beer,taste=taste,user=user, user_id=user_id, beer_id=beer_id, appdata = 1)
    except KeyError, IntegrityError:
        print 'user - beer pair already in db or empty field present'
        return
    print 'rating added successfully'

def add_pred_to_db(user_id, beer_id, pred, engine):
    metadata = MetaData(engine)
    preds = Table('predictions', metadata, autoload=True)
    i = preds.insert()
    i.execute( user_id=user_id, beer_id=beer_id, pred = pred )





if __name__ == '__main__':
    # set up environment
    conf = SparkConf() \
      .setAppName("BeerSleuthALS") \
      .set("spark.executor.memory", "4g")
    sc = SparkContext(conf=conf)
    sqlContext = SQLContext(sc) 

    #load data
    engine = create_engine('postgresql://postgres:123@localhost:5432/beersleuth')
    ratings_sqldf = get_item_user_rev_from_pg(engine, sqlContext)
    sqlContext.registerDataFrameAsTable(ratings_sqldf, "ratings")
#    train, test = sqlContext.table('ratings').randomSplit([.8, .2])
#    train = train.cache()
#    test = test.cache()
##    add_rating_to_db(user='johnjohn', beer=u'101 North Heroine IPA' , taste=8, engine=engine)
##    add_rating_to_db(user='johnjohn', beer=u'Boulder Creek Golden Promise' , taste=6, engine=engine)
##    model_param_sweep(train, test)
#    import timeit
#    start_time = timeit.default_timer()
    model = fit_final_model(ratings_sqldf)
#    elapsed = timeit.default_timer() - start_time
'''
initial CA ratings db had 627431 ratings
CREATE TABLE beercounts AS (SELECT min(beer) beer ,min(beer_id) beer_id, count(*) FROM mt3ratings group by beer, beer_id);

'''     