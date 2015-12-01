from pymongo import MongoClient
from sqlalchemy import create_engine
import psycopg2
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import datetime

import cPickle as pickle

client = MongoClient()
db = client['ratebeer']
beer_ratings = db.ratings


def create_sql_tables(df, table_name):
    engine = create_engine('postgresql://postgres:123@localhost:5432/beersleuth')
    df.to_sql(table_name, engine)


def get_item_user_pairs(beer_ratings):
    taste_df = pd.DataFrame(list(beer_ratings.find({},{'beer':1,'taste':1,'user':1}))).sort_values(by='user')
    user_df = pd.DataFrame(list(beer_ratings.distinct('user')),columns=['user'])
    user_df['user_id']=user_df.index
    beer_df = pd.DataFrame(list(beer_ratings.distinct('beer')),columns=['beer'])
    beer_df['beer_id']=beer_df.index
    taste_df = taste_df.merge(user_df, how='left').merge(beer_df, how='left')
    taste_df = taste_df[taste_df.taste != 'NA']
    taste_df.taste = taste_df.taste.astype(float)*10
    create_sql_tables(taste_df, 'ratings')

def get_CA_item_user_pairs(beer_ratings):
    taste_df = pd.DataFrame(list(beer_ratings.find({'region':'United States: California'},{'beer':1,'taste':1,'user':1}))).sort_values(by='user')
    user_df = pd.DataFrame(list(beer_ratings.distinct('user')),columns=['user'])
    user_df['user_id']=user_df.index
    beer_df = pd.DataFrame(list(beer_ratings.distinct('beer')),columns=['beer'])
    beer_df['beer_id']=beer_df.index
    taste_df = taste_df.merge(user_df, how='left').merge(beer_df, how='left')
    taste_df = taste_df[taste_df.taste != 'NA']
    taste_df.taste = taste_df.taste.astype(float)*10
    create_sql_tables(taste_df, 'caratings')


def get_beer_side_data(beer_ratings):
    beer_side_data = pd.DataFrame(list(beer_ratings.find({},{"beer":1, "abv":1, "calories":1, 'brewery': 1, 'ratebeer_rating': 1, 'style': 1, 'region': 1})))
    beer_side_data.pop('_id')
    beer_side_data.drop_duplicates(inplace=True)
    beer_side_data['beer_id'] = beer_side_data.index
    abv_mode = beer_side_data[beer_side_data['abv'] != 'NA']['abv'].mode().values[0]
    beer_side_data['abv'] =  beer_side_data['abv'].replace('NA', abv_mode)
    beer_side_data['abv'] =  beer_side_data['abv'].apply(lambda x: x.replace('%', ''))
    beer_side_data['abv'] =  pd.to_numeric(beer_side_data['abv'], errors='coerce')
    cal_mode = beer_side_data[beer_side_data['calories'] != 'NA']['calories'].mode().values[0]
    beer_side_data['calories'] =  beer_side_data['calories'].replace('NA', cal_mode)
    beer_side_data['calories'] =  pd.to_numeric(beer_side_data['calories'], errors='coerce')
    beer_side_data['ratebeer_rating'] =  pd.to_numeric(beer_side_data['ratebeer_rating'], errors='coerce')
    create_sql_tables(beer_side_data, 'beers')


def nlp(beer_ratings):
  #  text_revs = pd.DataFrame(list(beer_ratings.find({},{"beer":1, 'review': 1})))
  #  text_revs.pop('_id')
  #  agged_text = text_revs.groupby('beer').aggregate(sum)
 #   with open('agged_text.pkl','wb') as f:
#            pickle.dump(agged_text, f)
    with open('agged_text.pkl','rb') as f:
        agged_text = pickle.load(f)
    tokenized_text_list = process_text(agged_text)
    return tokenized_text_list

def process_text(df):
    wn_tokens = []
    stop_word_set = set(stopwords.words('english'))
    wordnet = WordNetLemmatizer()
#    snowball = SnowballStemmer('english')
    for row in xrange(len(df)):
        print(row)
        string =df.ix[row].values[0].lower()
        temp_list =word_tokenize(string)
        temp_list = filter( lambda word: word not in stop_word_set, temp_list)
        final_list = [wordnet.lemmatize(word) for word in temp_list]
#        final_list = [snowball.stem(word) for word in temp_list]
        wn_tokens.append(" ".join(final_list))
    return wn_tokens

def calc_term_freq():
    with open('tokens.pkl','rb') as f:
        tokens = pickle.load(f)
    with open('agged_text.pkl','rb') as f:
            agged_text = pickle.load(f)
    v = TfidfVectorizer(max_features=100, ngram_range=(1, 2), analyzer = 'word', stop_words=['10', '12'], use_idf=False)
    tf = v.fit_transform(tokens)
    tf = tf.todense()
    text_df = pd.DataFrame(tf, index = agged_text.index, columns = v.vocabulary_)
    text_df.pop('beer')
    create_sql_tables(text_df, 'text_data')
    return
