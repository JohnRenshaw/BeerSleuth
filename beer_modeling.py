import graphlab as gl
import pandas as pd
from beer_munging import get_item_user_pairs, get_beer_side_data
from pymongo import MongoClient

def param_sweep(sframe ):
    '''
    to use ...

    from GS import param_sweep
    job = param_sweep(ratings_sdf)

    to see results of param_sweep use job.get_results() 
    '''
    train, valid = gl.recommender.util.random_split_by_user(taste_sf, user_id='user', item_id='beer', max_num_users=4000, item_test_proportion=.2)
    params = {      'target': 'taste',
 #                   'user_data': user_side,
 #                   'item_data': movie_side,
                    'user_id':'user',
                    'item_id':'beer',
                    'num_factors': [8],
                    'regularization': [1e-3, 1e-4, 1e-5, 1e-6],
                    'linear_regularization':1e-10,
                    'side_data_factorization': False,
                    'nmf' : False, 
                    'max_iterations': 60,
                    'sgd_step_size': 0,
                    'solver':'auto',
                    'verbose':True
                    }

    gs = gl.grid_search.create((train, valid), gl.recommender.factorization_recommender.create, params)
    return gs

if __name__ == '__main__':
    client = MongoClient()
    db = client['ratebeer']
    beer_ratings = db.ratings
    taste_df = get_item_user_pairs(beer_ratings)
    beer_sides_sf = gl.SFrame(get_beer_side_data(beer_ratings))
    taste_sf = gl.SFrame(taste_df)
  #  train, valid = gl.recommender.util.random_split_by_user(taste_sf, user_id='user', item_id='beer', max_num_users=4000, item_test_proportion=.2)
  #  m = gl.recommender.factorization_recommender.create(train, user_id='user', item_id='beer', \
   #                      item_data=beer_sides_sf,\
  #                       num_factors=8, target='taste', max_iterations=50)
 #   preds = m.predict(valid)
    gs = param_sweep(taste_sf)
