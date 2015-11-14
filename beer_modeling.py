from collections import Counter
import graphlab as gl
import pandas as pd
import os
import numpy as np
from sqlalchemy import create_engine
import psycopg2


gl.set_runtime_config('GRAPHLAB_DEFAULT_NUM_PYLAMBDA_WORKERS', 36)

def param_sweep(taste_sf, beer_sf ):
    '''
    to use ...

    from GS import param_sweep
    job = param_sweep(ratings_sdf)

    to see results of param_sweep use job.get_results() 
    '''
    train, valid = gl.recommender.util.random_split_by_user(taste_sf, user_id='user',\
         item_id='beer', max_num_users=4000, item_test_proportion=.2)
    params = dict([('user_id','user'),
                    ('item_id','beer'),
                    ('target', 'taste'),
                    ('regularization',[1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]),
                    ('linear_regularization',[1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]),
		    ('side_data_factorization', False) ])
    rs = gl.random_search.create(datasets=(train, valid), model_factory=gl.recommender.factorization_recommender.create, model_parameters=params, max_models=25)
    return rs


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
    taste_df.to_csv('taste_df_CA.csv', encoding = 'utf-8')   

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
        linear_regularization=1e-4,
        nmf=False, 
        num_factors=8, 
        target='taste',
        max_iterations=50,
        sgd_step_size=0,
        side_data_factorization=True)
    return m, train, valid


def load_sframes_from_s3():
    beer_sf = gl.SFrame('https://s3-us-west-2.amazonaws.com/beerdata/beer_df_wout_brewery.csv')
    taste_sf = gl.SFrame('https://s3-us-west-2.amazonaws.com/beerdata/taste_df_CA.csv')
    beer_sf = beer_sf.dropna()
    taste_sf = taste_sf.dropna()
    return taste_sf, beer_sf

def load_sframes_from_csv():
    beer_sf = gl.SFrame(data='beer_df_wout_brewery.csv')
    beer_sf = beer_sf.dropna()
    taste_sf = gl.SFrame(data='taste_df_CA.csv')
    taste_sf = taste_sf.dropna()
    return taste_sf, beer_sf

def error_breakdown(truth, pred):
    '''
    use this function to get a datafram of the errors grouped by the actual value
    '''
    error = pd.DataFrame(np.array(pred - truth))
    truth = pd.DataFrame(np.array(truth))
    comb = pd.concat([truth, error], axis=1)
    comb.columns = ['act', 'error']
    comb = comb.sort_values(by='act')
    return comb

def classification_rate(m, train, test):
    """
    custom function to score model, rule is as follows:
    if true_score < 5 and pred < 5 : match
    if true_score > 5 and error < 1 : match
    others not considered a match
    """
    train_preds = m.predict(train)
    test_preds = m.predict(test)
    train_error=np.array(np.abs(train_preds - train['taste']))
    test_error=np.array(np.abs(test_preds - test['taste']))
    train_biased_recall = np.zeros(train_error.shape[0])
    test_biased_recall = np.zeros(test_error.shape[0])
    train_biased_recall[train['taste'] > 5 and train_error < 1] = 1
    train_biased_recall[train['taste'] <= 5 and train_preds <= 5] = 1
    test_biased_recall[test['taste'] > 5 and test_error < 1] = 1
    test_biased_recall[test['taste'] <= 5 and test_preds <= 5] = 1
    train_biased_recall_score = np.sum(train_biased_recall)/train_biased_recall.shape[0]
    test_biased_recall_score = np.sum(test_biased_recall)/test_biased_recall.shape[0]
    score_dict = dict([('train_classification_rate', train_biased_recall_score), ('test_classification_rate', test_biased_recall_score)])
    return score_dict


def custom_random_search(taste_sf, beer_sf, reg_list, lin_reg_list, nmf_list, num_factors_list, max_iterations_list, num_models, side_list):
    '''
     random parameter seach, wrote because graphlabs wasnt working with side data, this runs in serial, better to use built in model if possible
     
     rand_search_results = custom_random_search(taste_sf, beer_sf, reg_list=np.logspace(0,-10),lin_reg_list=np.logspace(0,-10),nmf_list= [True, False],\
     num_factors_list=[8,12,16], max_iterations_list=[50], num_models=10, side_list=False) 
    '''
    train, valid = gl.recommender.util.random_split_by_user(taste_sf, user_id='user', item_id='beer', max_num_users=4000, item_test_proportion=.2)
    agg = [] 
    for i in xrange(num_models):
        this_reg = np.random.choice(reg_list)
        this_lin_reg = np.random.choice(lin_reg_list)
        this_nmf = np.random.choice(nmf_list)
        this_num_factors= np.random.choice(num_factors_list)
        this_max_iterations = np.random.choice(max_iterations_list)
        this_side_data_fact = np.random.choice(side_list)
        m = gl.recommender.factorization_recommender.create(train,
            user_id='user',
            item_id='beer',
            item_data=beer_sf, 
            regularization=this_reg,
            linear_regularization=this_lin_reg,
            nmf=this_nmf, 
            num_factors=this_num_factors, 
            target='taste',
            max_iterations=this_max_iterations,
            sgd_step_size=0,
            side_data_factorization=this_side_data_fact)
        class_rate = classification_rate(m, train, valid)
        this_train_class_rate = class_rate['train_classification_rate']
        this_test_class_rate = class_rate['test_classification_rate']
        score = m.evaluate(valid)
        test_rmse = score['rmse_overall']
        agg.append([this_reg, this_lin_reg, this_nmf, this_max_iterations, this_num_factors, this_side_data_fact, m.get('training_rmse'), test_rmse, this_train_class_rate, this_test_class_rate ])
    agg_df = pd.DataFrame(agg, columns = ['reg', 'lin_reg', 'nmf', 'max_iterations', 'num_factors', 'side_data_factorization', 'training_rmse', 'test_rmse', 'train_class_rate', 'test_class_rate'])
    agg_df.to_csv('random_search.csv')
    return agg_df

