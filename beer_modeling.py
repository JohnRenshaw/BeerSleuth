from collections import Counter
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
                    'num_factors': [ 8, 12, 16, 20],
                    'regularization': [1e-2,1e-4, 1e-6, 1e-8],
                    'linear_regularization':[1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
                    'side_data_factorization': False,
                    'nmf' : False, 
                    'max_iterations': 40,
                    'sgd_step_size': 0,
                    'solver':'auto',
                    'verbose':True
                    }

    gs = gl.random_search.create((train, valid), gl.recommender.factorization_recommender.create, params, max_models=20)
    return gs

def load_data_from_sql():
    engine = create_engine('postgresql://postgres:123@localhost:5432/beersleuth')
    taste_df = pd.read_sql_query('''
        SELECT* FROM caratings
        WHERE caratings.user NOT IN
            (SELECT counts.reviewer FROM 
                (SELECT caratings.user as reviewer,count(*) FROM caratings GROUP BY caratings.user) as counts 
                WHERE counts.count < 4)
         ''', engine)
    beer_df = pd.read_sql_table('beers', engine)
    text_df = pd.read_sql_table('text_rev_data', engine)
    text_df.pop('index')
    text_df.pop('beer')
    return taste_df, beer_df, text_df

def rem_unicode_from_colnames(colnames):
    new_colnames = map(lambda x:x.encode('ascii','ignore'), colnames)
    return new_colnames


if __name__ == '__main__':
    #load  data
    taste_df, beer_df, text_df = load_data_from_sql()

    #feature engineering - style
    style_counts = Counter(beer_df.style)
    top_x_styles = [x[0] for x in style_counts.most_common(30)]
    style_dummies = pd.get_dummies(beer_df.style)
    style_dummies = style_dummies[top_x_styles]
    style_dummies.columns = rem_unicode_from_colnames(style_dummies.columns)
    beer_df = pd.concat([beer_df, style_dummies, text_df], axis=1)
    beer_df.pop('style')
 #   beer_df = pd.merge(beer_df, text_df)


#    #feature engineering - brewery
#    beer_sf = gl.SFrame(beer_df)
#    brewery_dummies = pd.get_dummies(beer_df.brewery, prefix='brewery')
#    brewery_dummies.columns = rem_unicode_from_colnames(brewery_dummies.columns)
#    beer_df = pd.concat([beer_df, brewery_dummies], axis=1)
#    beer_df.pop('brewery')
    
    beer_sf = gl.SFrame(beer_df)
    beer_sf = beer_sf.dropna()
    taste_sf = gl.SFrame(taste_df)
    train, valid = gl.recommender.util.random_split_by_user(taste_sf, user_id='user', item_id='beer', max_num_users=4000, item_test_proportion=.2)
    m = gl.recommender.factorization_recommender.create(train,
        user_id='user',
        item_id='beer',
        item_data=beer_sf, 
        regularization=1e-6,
        linear_regularization=1e-8,
        nmf=False, 
        num_factors=12, 
        target='taste',
        max_iterations=50,
        sgd_step_size=0.0039,
        side_data_factorization=True)
    preds = m.predict(valid)
#    gs = param_sweep(taste_sf)


