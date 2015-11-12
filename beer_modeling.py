from collections import Counter
import graphlab as gl
import pandas as pd
import os
import numpy as np
#from sqlalchemy import create_engine
#import psycopg2

def param_sweep(taste_sf, beer_sf ):
    '''
    to use ...

    from GS import param_sweep
    job = param_sweep(ratings_sdf)

    to see results of param_sweep use job.get_results() 
    '''
    train, valid = gl.recommender.util.random_split_by_user(taste_sf, user_id='user',\
         item_id='beer', max_num_users=4000, item_test_proportion=.2)
    params = {      'target': 'taste',
                    'user_id':'user',
                    'item_id':'beer',
 #                  'user_data': user_side,
                    'item_data': [beer_sf],
                    'num_factors': [8, 12],
                    'regularization': [1e-2, 1e-4, 1e-6, 1e-8, 1e-10],
                    'linear_regularization':[1e-2,1e-4, 1e-6, 1e-8, 1e-10],
                    'side_data_factorization': [True, False],
                    'nmf' : [True, False], 
                    'max_iterations': 50,
                    'sgd_step_size': 0,
                    'solver':'auto',
                    'verbose':True
                    }

    gs = gl.random_search.create((train, valid), gl.recommender.factorization_recommender.create, params, max_models=15, perform_trial_run=False )
    return gs


def load_data_from_sql():
    engine = create_engine('postgresql://postgres:123@localhost:5432/beersleuth')
    taste_df = pd.read_sql_query('''
        SELECT * FROM ratings
        WHERE ratings.user NOT IN
            (SELECT counts.reviewer FROM 
                (SELECT ratings.user as reviewer,count(*) FROM ratings GROUP BY ratings.user) as counts 
                WHERE counts.count < 4)
         ''', engine)

    beer_df = pd.read_sql_table('beers', engine)
    beer_df.index=beer_df.beer
    beer_df.pop('beer')
    text_df = pd.read_sql_table('text_rev_data', engine)
    text_df.index=text_df.beer
    text_df.pop('beer')
    return taste_df, beer_df, text_df

def rem_unicode_from_colnames(colnames):
    new_colnames = map(lambda x:x.encode('ascii','ignore'), colnames)
    return new_colnames


def write_csv():
    #load  data
    taste_df, beer_df, text_df = load_data_from_sql()

    #feature engineering - style
    style_counts = Counter(beer_df.style)
    top_x_styles = [x[0] for x in style_counts.most_common(30)]
    style_dummies = pd.get_dummies(beer_df.style)
    style_dummies = style_dummies[top_x_styles]
    style_dummies.columns = rem_unicode_from_colnames(style_dummies.columns)
    beer_df = pd.merge(beer_df, text_df, left_index=True, right_index=True)
    beer_df = pd.merge(beer_df, style_dummies, left_index=True, right_index=True)
    beer_df['beer'] = beer_df.index
    beer_df.pop('index')
    beer_df.pop('style')
#   beer_df = create_brewery_dummies(beer_df) 
    taste_df.to_csv('taste_df_corpus.csv', encoding = 'utf-8')   

def create_brewery_dummies(beer_df):
    copy_beer_df = beer_df
    brewery_dummies = pd.get_dummies(copy_beer_df.brewery, prefix='brewery')
    brewery_dummies.columns = rem_unicode_from_colnames(brewery_dummies.columns)
    copy_beer_df = pd.merge(copy_beer_df, brewery_dummies, left_index=True, right_index=True)
    copy_beer_df.pop('brewery')
    return copy_beer_df

def start_cluster():
    my_config = gl.deploy.Ec2Config(instance_type = 'c3.2xlarge', region = 'us-west-2')
    my_cluster = gl.deploy.ec2_cluster.create('Compute Cluster',
                                            's3://beerdata',
                                            my_config,
                                            num_hosts = 5)
    return my_cluster

def fit_model(taste_sf, beer_sf):
    train, valid = gl.recommender.util.random_split_by_user(taste_sf, user_id='user', item_id='beer', max_num_users=4000, item_test_proportion=.2)
    m = gl.recommender.factorization_recommender.create(train,
        user_id='user',
        item_id='beer',
        item_data=beer_sf, 
        regularization=1e-3,
        linear_regularization=1e-5,
        nmf=True, 
        num_factors=12, 
        target='taste',
        max_iterations=50,
        sgd_step_size=0,
        side_data_factorization=False)
    return m


def load_sframes_from_s3():
    beer_sf = gl.SFrame('s3://beerdata/beer_df_wout_brewery.csv')
    taste_sf = gl.SFrame('s3://beerdata/taste_df.csv')
    beer_sf = beer_sf.dropna()
    return beer_sf, taste_sf

def load_sframes_from_csv():
    beer_sf = gl.SFrame(data='beer_df.csv')
    beer_sf = beer_sf.dropna()
    taste_sf = gl.SFrame(data='taste_df.csv')
    return beer_sf, taste_sf

