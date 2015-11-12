def workflow():
    taste_df, beer_sf = get_data()
    gs = param_sweep(taste_sf, beer_sf)
    return gs


def get_data():
    """
    loads data from s3 instance
    """
    import graphlab as gl 
    beer_sf = gl.SFrame.read_csv('beer_df.csv')
    beer_sf = beer_sf.dropna()
    taste_sf = gl.SFrame.read_csv('taste_df_corpus.csv')
    return taste_sf, beer_sf


def fit_model(taste_sf, beer_sf):
    """
    fits factorization_recommender to beer-user pairs in taste_df and uses side data from beer_df
    """
    train, valid = gl.recommender.util.random_split_by_user(taste_sf, user_id='user', item_id='beer', max_num_users=4000, item_test_proportion=.2)
    m = gl.recommender.factorization_recommender.create(train,
        user_id='user',
        item_id='beer',
        item_data=beer_sf, 
        regularization=1e-3,
        linear_regularization=1e-5,
        nmf=False, 
        num_factors=12, 
        target='taste',
        max_iterations=50,
        sgd_step_size=0.005,
        side_data_factorization=False)
    return m

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

def start_cluster():
    import graphlab as gl
    my_config = gl.deploy.Ec2Config(instance_type = 'c3.2xlarge', region = 'us-east-1')
    my_cluster = gl.deploy.ec2_cluster.create('Compute Cluster',
                                            s3_path='s3://beerdata',
                                            ec2_config = my_config,
                                            num_hosts = 5)
    return my_cluster

'''
my_cluster = gl.deploy.ec2_cluster.load('s3://beerdata')
gl.aws.set_credentials(os.environ['AWS_ACCESS_KEY'],os.environ['AWS_SECRET_KEY'])

'''    