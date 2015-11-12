from collections import Counter
import graphlab as gl
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import psycopg2

def param_sweep(sframe, beer_sf ):
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
                    'nmf' : [True], 
                    'max_iterations': 50,
                    'sgd_step_size': 0,
                    'solver':'auto',
                    'verbose':True
                    }

    gs = gl.random_search.create((train, valid), gl.recommender.factorization_recommender.create, params, max_models=20, perform_trial_run=False )
    return gs

def load_data_from_sql():
    engine = create_engine('postgresql://postgres:123@localhost:5432/beersleuth')
    taste_df = pd.read_sql_query('''
        SELECT * FROM caratings
        WHERE caratings.user NOT IN
            (SELECT counts.reviewer FROM 
                (SELECT caratings.user as reviewer,count(*) FROM caratings GROUP BY caratings.user) as counts 
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

##below is the query to limit users AND beers for more than 3 
'''
        SELECT * FROM caratings
        WHERE caratings.user NOT IN
            (SELECT counts.reviewer FROM 
                (SELECT caratings.user as reviewer,count(*) FROM caratings GROUP BY caratings.user) as counts 
                WHERE counts.count < 4)
        AND caratings.beer NOT IN
                    (SELECT bcounts.beer FROM 
                (SELECT caratings.beer as beer,count(*) FROM caratings GROUP BY caratings.beer) as bcounts 
                WHERE bcounts.count < 4)
'''


if __name__ == '__main__':
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


#    #feature engineering - brewery
#    beer_sf = gl.SFrame(beer_df)
    brewery_dummies = pd.get_dummies(beer_df.brewery, prefix='brewery')
    brewery_dummies.columns = rem_unicode_from_colnames(brewery_dummies.columns)
    beer_df = pd.merge(beer_df, brewery_dummies, left_index=True, right_index=True)
    beer_df.pop('brewery')
    
    beer_sf = gl.SFrame(beer_df)
    beer_sf = beer_sf.dropna()
    taste_sf = gl.SFrame(taste_df)
#    train, valid = gl.recommender.util.random_split_by_user(taste_sf, user_id='user', item_id='beer', max_num_users=4000, item_test_proportion=.2)
#    m = gl.recommender.factorization_recommender.create(train,
#        user_id='user',
#        item_id='beer',
#        item_data=beer_sf, 
#        regularization=1e-3,
#        linear_regularization=1e-5,
#        nmf=False, 
#        num_factors=12, 
#        target='taste',
#        max_iterations=50,
#        sgd_step_size=0.005,
#        side_data_factorization=True)
#    preds = m.predict(valid)
    gs = param_sweep(taste_sf, beer_sf)


