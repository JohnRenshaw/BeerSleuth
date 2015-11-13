import graphlab as gl

def workflow():
    taste_sf, beer_sf = get_data()
    m= fit_model(taste_sf, beer_sf)
    return m


def get_data():
    """
    loads data from s3 instance
    """
    beer_sf = gl.SFrame.read_csv('https://s3-us-west-2.amazonaws.com/beerdata/beer_df.csv')
    beer_sf = beer_sf.dropna()
    taste_sf = gl.SFrame.read_csv('https://s3-us-west-2.amazonaws.com/beerdata/taste_df_corpus.csv')
    return taste_sf, beer_sf


def fit_model(taste_sf, beer_sf):
    """
    fits factorization_recommender to beer-user pairs in taste_df and uses side data from beer_df
    """
    train, valid = gl.recommender.util.random_split_by_user(taste_sf, user_id='user', item_id='beer', max_num_users=4000, item_test_proportion=.2)
    m = gl.recommender.ranking_factorization_recommender.create(train,
        user_id='user',
        item_id='beer',
        item_data=beer_sf, 
        regularization=1e-4,
        linear_regularization=1e-5,
        nmf=False, 
        num_factors=12, 
        target='taste',
        max_iterations=15,
        sgd_step_size=0,
        side_data_factorization=True)
    m.evaluate(valid)
    return m

def param_sweep(taste_sf, beer_sf):
    '''
    to use ...

    from GS import param_sweep
    job = param_sweep(ratings_sdf)

    to see results of param_sweep use job.get_results() 
    '''
    import graphlab as gl
    train, valid = gl.recommender.util.random_split_by_user(taste_sf, user_id='user',\
         item_id='beer', max_num_users=4000, item_test_proportion=.2)
    params = {      'target': 'taste',
                    'user_id':'user',
                    'item_id':'beer',
                    'item_data': [beer_sf],
                    'num_factors': [8, 12],
                    'regularization': [1, 1e-2, 1e-4, 1e-6, 1e-8, 1e-10],
                    'linear_regularization':[1, 1e-2,1e-4, 1e-6, 1e-8, 1e-10],
                    'side_data_factorization': [True, False],
                    'nmf' : [True, False], 
                    'max_iterations': 50,
                    'sgd_step_size': 0,
                    'solver':'auto',
                    'verbose':True
                    }

    gs = gl.random_search.create((train, valid), gl.recommender.factorization_recommender.create, params, max_models=20, perform_trial_run=True )
    return gs

def start_cluster():

    my_config = gl.deploy.Ec2Config(instance_type = 'c3.8xlarge',
     region = 'us-west-2')

    my_cluster = gl.deploy.ec2_cluster.create('Single',
                                            s3_path='s3://beerdata',
                                            ec2_config = my_config,
                                            num_hosts = 1)
    return my_cluster

def reconnect_cluster(s3_path):
    """
    simple method to reconnect to ec2 instances running graphlab
    use like 
    my_cluster = reconnect('s3://beerdata')
    """
    login()
    return gl.deploy.ec2_cluster.load(s3_path)

def login():
    import os
    import graphlab as gl
    gl.aws.set_credentials(os.environ['AWS_ACCESS_KEY'],os.environ['AWS_SECRET_KEY'])




'''
 job = gl.deploy.job.create(workflow, environment=my_cluster)
'''    