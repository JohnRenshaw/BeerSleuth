from pymongo import MongoClient
from sqlalchemy import create_engine
import pandas as pd
import datetime
import psycopg2

client = MongoClient()
db = client['ratebeer']
beer_ratings = db.ratings


def create_sql_tables(df, table_name):
    engine = create_engine('postgresql://postgres:123@localhost:5432/beersleuth')
    df.to_sql(table_name, engine)


def get_item_user_pairs(beer_ratings):
    taste_df = pd.DataFrame(list(beer_ratings.find({},{'beer':1,'taste':1,'user':1}))).sort_values(by='user')
    taste_df = taste_df[taste_df.taste != 'NA']
    taste_df.taste = taste_df.taste.astype(float)*5
    agged = taste_df.groupby('user').aggregate(len)
    drop_list = list(agged[ agged[ '_id'] < 4].index)
    taste_df = taste_df[-taste_df.user.isin(drop_list)]
    create_sql_tables(taste_df, 'ratings')
 

def get_beer_side_data(beer_ratings):
    beer_side_data = pd.DataFrame(list(beer_ratings.find({},{"beer":1, "abv":1, "calories":1, 'brewery': 1, 'ratebeer_rating': 1, 'style': 1})))
    beer_side_data.pop('_id')
    beer_side_data.drop_duplicates(inplace=True)
    abv_mode = beer_side_data[beer_side_data['abv'] != 'NA']['abv'].mode().values[0]
    beer_side_data['abv'] =  beer_side_data['abv'].replace('NA', abv_mode)
    beer_side_data['abv'] =  beer_side_data['abv'].apply(lambda x: x.replace('%', ''))
    beer_side_data['abv'] =  pd.to_numeric(beer_side_data['abv'], errors='coerce')
    cal_mode = beer_side_data[beer_side_data['calories'] != 'NA']['calories'].mode().values[0]
    beer_side_data['calories'] =  beer_side_data['calories'].replace('NA', cal_mode)
    beer_side_data['calories'] =  pd.to_numeric(beer_side_data['calories'], errors='coerce')
    beer_side_data['ratebeer_rating'] =  pd.to_numeric(beer_side_data['ratebeer_rating'], errors='coerce')
    create_sql_tables(beer_side_data, 'beers')

