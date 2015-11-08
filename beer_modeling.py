import graphlab as gl
import pandas as pd
from sqlalchemy import create_engine
import psycopg2

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
                    'num_factors': [8, 12],
                    'regularization': [1e-3, 1e-4],
                    'linear_regularization':[1e-8, 1e-6, 1e-4],
                    'side_data_factorization': False,
                    'nmf' : False, 
                    'max_iterations': 60,
                    'sgd_step_size': 0,
                    'solver':'auto',
                    'verbose':True
                    }

    gs = gl.random_search.create((train, valid), gl.recommender.factorization_recommender.create, params, max_models=10)
    return gs

def load_data_from_sql():
    engine = create_engine('postgresql://postgres:123@localhost:5432/beersleuth')
    taste_df = pd.read_sql_table('ratings', engine)
    beer_df = pd.read_sql_table('beers', engine)
    return taste_df, beer_df


if __name__ == '__main__':
    taste_df, beer_df = load_data_from_sql()
    beer_df.pop('ratebeer_rating')
    beer_sf = gl.SFrame(beer_df)
    taste_sf = gl.SFrame(taste_df)
#    train, valid = gl.recommender.util.random_split_by_user(taste_sf, user_id='user', item_id='beer', max_num_users=4000, item_test_proportion=.2)
#    m = gl.recommender.factorization_recommender.create(train, user_id='user', item_id='beer', \
#                         item_data=beer_sf, regularization=1e-4, linear_regularization=1e-6, \
#                         num_factors=8, target='taste', max_iterations=75)
 #   preds = m.predict(valid)
    gs = param_sweep(taste_sf)
