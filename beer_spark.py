
import sys
import itertools
import math
from operator import add
from os.path import join, isfile, dirname
from pyspark import SparkConf, SparkContext
from pyspark import SQLContext, HiveContext
from pyspark.ml.recommendation import ALS
import pandas as pd
import numpy as np
from pymongo import MongoClient
from sqlalchemy import create_engine
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

def get_item_user_rev_from_pg():
    engine = create_engine('postgresql://postgres:123@localhost:5432/beersleuth')
    taste_df = pd.read_sql_query('''
        SELECT * FROM caratings 
        WHERE caratings.user NOT IN 
            (SELECT counts.reviewer FROM  
                (SELECT caratings.user as reviewer,count(*) FROM caratings GROUP BY caratings.user)
                 as counts WHERE counts.count < 4)
         ''', engine)

    taste_df = taste_df[['user_id', 'beer_id', 'taste']]
    return taste_df

if __name__ == '__main__':
    # set up environment
    conf = SparkConf() \
      .setAppName("BeerSleuthALS") \
      .set("spark.executor.memory", "2g")
    sc = SparkContext(conf=conf)
    sqlContext = SQLContext(sc)


    #load data
    ratings_data = parseRating('ratings_spark.csv')
    pd_df = mongo_to_df()
    agged =pd_df.groupby('user_id', as_index=False).aggregate(len)
    keep_list = agged[agged.taste > 4].user_id
    ratings_sqldf = sqlContext.createDataFrame(pd_df[pd_df.user_id.isin(keep_list)])
  #  ratings_sqldf = sqlContext.createDataFrame(ratings_data, ['user_id', 'beer_id', 'taste'])
    sqlContext.registerDataFrameAsTable(ratings_sqldf, "ratings")
  #  smaller_sqldf = sqlContext.sql('SELECT * FROM ratings WHERE user NOT IN (SELECT reviewer FROM  (SELECT user as reviewer, count(*) FROM ratings GROUP BY user) counts WHERE count < 4)')

    #split into training and test
 #   training_RDD, test_RDD = ratings_data.randomSplit([8, 2], seed=0L)
 #   test_for_predict_RDD = test_RDD.map(lambda x: (x[0], x[1]))

    #model params
    iterations = 30
    regularization_param_list = np.linspace(0.15, 0.25, 5)

    #params used in keeping track of error between different ranks
    rank = 12
    errors = np.zeros(len(regularization_param_list))
    err = 0
    min_error = float('inf')
    best_rank = -1
    best_iteration = -1
    als = ALS().setItemCol('beer_id').setUserCol('user_id').setRatingCol('taste')
    model = als.fit(ratings_sqldf)
#    for reg in regularization_param_list:
#        model = ALS.train(ratings_sqldf, userCol='user_id', item_col='beer_id',ratingCol='taste',  rank, iterations=iterations, lambda_=reg)
#        predictions = model.predictAll(test_for_predict_RDD).map(lambda r: ((r[0], r[1]), r[2]))
#        rates_and_preds = test_RDD.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
#        error = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean())
#        errors[err] = error
#        err += 1
#        print 'For regParam %s the RMSE is %s' % (reg, error)
#        if error < min_error:
#            min_error = error
#            best_rank = reg#

#    print 'The best model was trained with regParam %s' % best_rank