from pymongo import MongoClient
from sqlalchemy import create_engine
import pandas as pd
import datetime
import psycopg2

client = MongoClient()
db = client['ratebeer']
beer_ratings = db.ratings
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
#reviews_df = ratings_df[['user','beer','combined']]
beer_df = ratings_df[['beer','style','brewery','beer_decr','abv','region','calories', 'ratebeer_rating']].drop_duplicates()
beer_df.index = beer_df.beer
beer_df.pop('beer')
#ratings_df.drop(['brewery','beer_decr','abv','region','calories','style','ratebeer_rating'],inplace=True,axis=1)

def create_sql_tables(df):
    engine = create_engine('postgresql://postgres:123@localhost:5432/beersleuth')
    df.to_sql("all_data", engine)


'''
CREATE TABLE beers AS SELECT DISTINCT beer, brewery, beer_decr,ratebeer_rating, region, style, abv, calories FROM all_data;

'''