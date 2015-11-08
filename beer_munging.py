from pymongo import MongoClient
from sqlalchemy import create_engine
import pandas as pd
import datetime
import psycopg2

client = MongoClient()
db = client['ratebeer']
beer_ratings = db.ratings

def clean_data(beer_ratings):
    ratings_df = pd.DataFrame(list(beer_ratings.find()))
    ratings_df.index = ratings_df['_id']
    ratings_df.pop('_id')
    ratings_df['abv'] = ratings_df['abv'].apply(lambda x:x.replace('%',''))
    ratings_df['abv'] = pd.to_numeric(ratings_df['abv'],errors='coerce')
    ratings_df['beer_decr'] = ratings_df['beer_decr'].apply(lambda x:x.replace('COMMERCIAL DESCRIPTION','').strip())
    ratings_df['appearance'] = pd.to_numeric(ratings_df['appearance'],errors='coerce')
    ratings_df['aroma'] = pd.to_numeric(ratings_df['aroma'],errors='coerce')
    ratings_df['calories'] = pd.to_numeric(ratings_df['calories'],errors='coerce')
    ratings_df['combined'] = pd.to_numeric(ratings_df['combined'],errors='coerce')
    ratings_df['overall'] = pd.to_numeric(ratings_df['overall'],errors='coerce')
    ratings_df['palate'] = pd.to_numeric(ratings_df['palate'],errors='coerce')
    ratings_df['ratebeer_rating'] = pd.to_numeric(ratings_df['ratebeer_rating'],errors='coerce')
    ratings_df['taste'] = pd.to_numeric(ratings_df['taste'],errors='coerce')
    ratings_df['date'] = ratings_df['date'].apply(lambda x: x.replace('does not count','').strip().replace('NA',''))
    ratings_df['date'] = pd.to_datetime(ratings_df['date'], format='%b %d, %Y', errors='coerce')
    beer_df = ratings_df[['beer','style','brewery','beer_decr','abv','region','calories', 'ratebeer_rating']].drop_duplicates()
    beer_df.index = beer_df.beer
    beer_df.pop('beer')


def create_sql_tables(df):
    engine = create_engine('postgresql://postgres:123@localhost:5432/beersleuth')
    df.to_sql("all_data", engine)

def get_item_user_pairs(beer_ratings):
    taste_df = pd.DataFrame(list(beer_ratings.find({},{'beer':1,'taste':1,'user':1}))).sort_values(by='user')
    taste_df = taste_df[taste_df.taste != 'NA']
    taste_df.taste = taste_df.taste.astype(float)*5
    agged = taste_df.groupby('user').aggregate(len)
    drop_list = list(agged[ agged[ '_id'] < 3].index)
    taste_df = taste_df[-taste_df.user.isin(drop_list)]
    return taste_df

def get_beer_side_data(beer_ratings):
    beer_side_data = pd.DataFrame(list(beer_ratings.find({},{"beer":1, "abv":1, "calories":1, 'brewery': 1, 'ratebeer_rating': 1, 'style': 1})))
    beer_side_data.pop('_id')
    beer_side_data.drop_duplicates(inplace=True)
    abv_mode = beer_side_data[beer_side_data['abv'] != 'NA']['abv'].mode().values[0]
    beer_side_data['abv'] =  beer_side_data['abv'].replace('NA', abv_mode)
    beer_side_data['abv'] =  beer_side_data['abv'].apply(lambda x: x.replace('%', '')).astype(float)
    return beer_side_data

'''
CREATE TABLE beers AS SELECT DISTINCT beer, brewery, beer_decr,ratebeer_rating, region, style, abv, calories FROM all_data;
beer_side_data = pd.DataFrame(list(beer_ratings.find({},{"beer":1, 'abv':1, "calories":1})))
beer_side_data.pop('_id')
beer_ratings.drop_duplicates(inplace=True)

'''

