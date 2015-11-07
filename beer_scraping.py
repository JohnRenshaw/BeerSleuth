import sys
import re
import requests
from bs4 import BeautifulSoup
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError
from fake_useragent import UserAgent
import threading
import time

p = re.compile(ur'([0-9]\/[0-9]*)')




def frac_string_to_dec(string):
    return float(string.split('/')[0])/float(string.split('/')[1])

def get_state_reviews(url, start_index):
    brewery_links = get_state_breweries(url)
    num_breweries = len(brewery_links)
    for i, brewery in enumerate(brewery_links):
        if i >= start_index:
            print('\n working on brewery %i of %i in state, from link - '+ url)%(i+1, num_breweries)
            beer_links = get_brewery_beer_list(brewery)
            if not beer_links: continue
            num_beers = len(beer_links)
            for b, beer in enumerate(beer_links):
                print beer
                print('\n working on beer %i of %i in brewery - ' + beer + '\n')%(b+1, num_beers)
                get_reviews(beer)


def get_state_breweries(url):
    ua = UserAgent()
    page = requests.get(url, headers= { "User-agent": ua.random})
    soup = BeautifulSoup(page.content, 'lxml')
    links = soup.select('a')
    links = filter(lambda x: '/brewers/' in x['href'], links)
    links = map(lambda x: 'http://www.ratebeer.com' + x['href'], links)
    return links

def get_brewery_beer_list(link):
    ua = UserAgent()
    beer_list = set(beer_ratings.distinct('beer'))
    page = requests.get(link, headers= {"User-agent": ua.random})
    soup = BeautifulSoup(page.content, 'lxml')
    try: beer_pages = int(soup.select('a.ballno')[-1].text)
    except IndexError:beer_pages = 1
    link_coll=[]
    for rpage in xrange(beer_pages):
        time.sleep(0.5)
        current_page = requests.get(link+'0/'+str(rpage+1)+'/')
        soup = BeautifulSoup(current_page.content, 'lxml')
        links = soup.select('tr td a')
        links = filter(lambda x: x['href'].startswith('/beer/'), links)
        links = filter(lambda x: '/rate/' not in x['href'], links)
        links = filter(lambda x: '/top-50/' not in x['href'], links)
        links = filter(lambda x: x.text not in beer_list, links)
        links = map(lambda x: 'http://www.ratebeer.com' + x['href'], links)
        link_coll.extend(links)
    return link_coll


def get_recent_beers(base_link):
    ua = UserAgent()
    page = requests.get(base_link, headers= { "User-agent": ua.random})
    soup = BeautifulSoup(page.content, 'lxml')
    links = soup.select('.bubble big a')
    links = filter(lambda x: '/beer/' in x['href'], links)
    links = map(lambda x: 'http://www.ratebeer.com' + x['href'], links)
    return links

def get_top50_beers():
    ua = UserAgent()
    url = 'http://www.ratebeer.com/beer/top-50/'
    page = requests.get(url, headers= {"User-agent": ua.random})
    soup = BeautifulSoup(page.content, 'lxml')
    links = soup.select('tr td a')
    links = filter(lambda x: x['href'].startswith('/beer/'), links)
    links = filter(lambda x: '/rate/' not in x['href'], links)
    links = filter(lambda x: '/top-50' not in x['href'], links)
    links = filter(lambda x: '/statistics/' not in x['href'], links)
    links = map(lambda x: 'http://www.ratebeer.com' + x['href'], links)
    for link in links:
        get_reviews(link)


def get_recent_reviews():
    for page in xrange(1, 100):
        base_link = 'http://www.ratebeer.com/beer-ratings/'
        if page == 1: links = get_recent_beers(base_link)
        else: links = get_recent_beers(base_link+'0/'+str(page)+'/')
        for i in xrange(15):
            print('\nworking on beer %s from recent reviews\n')%links[i]
            get_reviews( links[i])


def get_reviews(beer_link):
    ua = UserAgent()
    page = requests.get(beer_link, headers= {   "User-agent": ua.random})
    soup = BeautifulSoup(page.content, 'lxml')
    metrics = soup.select('table tr td div tr strong')
    try: abv = soup.findAll('strong',text =re.compile("%"))[0].text
    except IndexError: abv = 'NA'
    try: style = soup.select('big a')[0].next_element.next_element.next_element.next_element.text
    except IndexError: style = 'NA'
    try: region = soup.findAll(attrs = {"itemprop":"title"})[1].text
    except IndexError: region = 'NA'
    try: beer_descr = soup.findAll('div', attrs={"style":"border: 1px solid #e0e0e0; background: #fff; padding: 14px; color: #777;"})[0].text
    except IndexError: beer_descr = 'NA'
    try: cals = soup.findAll('abbr',attrs={"title":"Estimated calories for a 12 fluid ounce serving"})[0].next_element.next_element.next_element.text
    except IndexError: cals = 'NA'
    try: title = soup.select('h1')[0].string
    except IndexError: return
    beer_list = beer_ratings.distinct("beer")
    if title in beer_list: return
    try:
        rating_text = soup.findAll(attrs = {"style":"font-size: 48px; font-weight: bold; color: #fff; padding: 7px 10px;"})[0]['title']
        ratebeer_rating = re.match(ur'(^[0-9.]+)', rating_text).group(0)
    except (IndexError, AttributeError): ratebeer_rating = 'NA'
    try :region = soup.findAll(attrs = {"itemprop":"title"})[1].text
    except IndexError: region = 'NA'
    try: review_pages = int(soup.select('a.ballno')[-1].text)
    except IndexError:review_pages = 1
    try: brewery = soup.select('big b a')[0].text
    except IndexError: return
    num_reviews = soup.select('div small big b span')[0].text
    if num_reviews == '0':
            write_to_mongo(title, "NA", style, beer_descr, ratebeer_rating, brewery, region,
                abv, cals, 'NA', "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA")
            return
    for rpage in xrange(review_pages):
        time.sleep(0.5)
        page = requests.get(beer_link+'1/'+str(rpage+1)+'/')
        soup = BeautifulSoup(page.content, 'lxml')
        metrics = soup.select('table tr td div tr strong')
        for i in xrange(len(metrics)):
            sys.stdout.write('.')
            try: review = soup.findAll('div',attrs={"style":"padding: 20px 10px 20px 0px; border-bottom: 1px solid #e0e0e0; line-height: 1.5;"})[i].text
            except IndexError: review = 'NA' 
            test = re.findall(p, str(metrics[i]))
            combined = float(metrics[i].previous_element)
            aroma = frac_string_to_dec(test[0])
            appearance = frac_string_to_dec(test[1])
            taste = frac_string_to_dec(test[2])
            palate = frac_string_to_dec(test[3])
            overall = frac_string_to_dec(test[4])
            user_info = soup.select('table tr td div tr td small a')[i].parent.text
            user = user_info.split(' - ')[0]
            location = user_info.split(' - ')[1]
            date = user_info.split(' - ')[2]
            write_to_mongo(title, user, style, beer_descr, ratebeer_rating, brewery, region,
                abv, cals, location, date, aroma, appearance, taste, palate, overall, combined, review)

        	
def write_to_mongo(title, user, style, beer_descr, ratebeer_rating, brewery, region,
    abv, cals, location, date, aroma, appearance, taste, palate, overall, combined, review):
    sys.stdout.write('.')
    review = {
            '_id':title+'_'+user,
            'beer': title,
            'style' : style,
            'beer_decr':beer_descr,
            'ratebeer_rating' : ratebeer_rating,
            'brewery': brewery,
            'region': region,
            'abv' : abv,
            'calories': cals,
            'user': user,
            'location': location,
            'date': date,
            'aroma': aroma,
            'appearance': appearance,
            'taste': taste,
            'palate': palate,
            'overall': overall,
            'combined':combined,
            'review' : review
            }
    try: beer_ratings.insert_one(review).inserted_id
    except DuplicateKeyError: print "tried to add duplicate"
    return

if __name__ == '__main__':
    client = MongoClient()
    db = client['ratebeer']
    beer_ratings = db.ratings
    #get_top50_beers() 
 #   get_recent_reviews()
#    jobs = []
#    job1 =threading.Thread(target=get_state_reviews, args=get_state_reviews('http://www.ratebeer.com/breweries/florida/9/213/', 0))
#    jobs.append(job1)
#    job1.start()
#    job2 = threading.Thread(target = get_recent_reviews, args = () )
#    jobs.append(job2)
#    job2.start()
    get_state_reviews('http://www.ratebeer.com/breweries/oregon/37/213/', 0)

'''
reminder to backup db - from cmf line
mongodump  --db ratebeer --collection ratings

    WA link http://www.ratebeer.com/breweries/washington/47/213/      done..
    OR link http://www.ratebeer.com/breweries/oregon/37/213/          done
    GA link http://www.ratebeer.com/breweries/georgia/10/213/         done
    CO link http://www.ratebeer.com/breweries/colorado/6/213/         dome
    CA link http://www.ratebeer.com/breweries/california/5/213/       done
    NV http://www.ratebeer.com/breweries/nevada/28/213/               done
    MA link http://www.ratebeer.com/breweries/massachusetts/21/213/   done
    AL likn http://www.ratebeer.com/breweries/alabama/1/213/          done
    FL    http://www.ratebeer.com/breweries/florida/9/213/            done

    other breweries
    http://www.ratebeer.com/brewers/caledonian-heineken-uk/168/

'''