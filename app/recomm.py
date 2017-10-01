import requests as rq
import StringIO
import zipfile
import pandas as pd
import time
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
class recomm_engine(object):
    def __init__(self):
        print 'initiating'
    
    def download_data(self,url):
        r = rq.get(url)
        z = zipfile.ZipFile(StringIO.StringIO(r.content))
        ratings = pd.read_csv(z.open('ml-latest-small/ratings.csv'))
        movies = pd.read_csv(z.open('ml-latest-small/movies.csv'))
        links = pd.read_csv(z.open('ml-latest-small/links.csv'))
        tags = pd.read_csv(z.open('ml-latest-small/tags.csv'))
        return ratings,movies,links,tags
    def train(self,n):
        start=time.time()
        self.ratings,self.movies,self.links,self.tags = self.download_data('http://files.grouplens.org/datasets/movielens/ml-latest-small.zip')
        print("Data ingested in %s seconds." % (time.time()-start))
        movies_df = pd.concat([self.movies,self.movies['genres'].str.get_dummies('|')],axis=1)
        movies_df = movies_df.drop('genres',1)
        movies_df.index=movies_df['movieId']
        cos_dist = pd.DataFrame(1-pairwise_distances(movies_df[movies_df.columns.difference(['movieId','title'])],metric='cosine'),index=movies_df['movieId'],columns=movies_df['movieId'])
        np.fill_diagonal(cos_dist.values, -1)
        arank = cos_dist.apply(np.argsort, axis=1)
        new_f=pd.DataFrame(arank.values[:,::-1][:,:n],index=cos_dist.index)
        dict_list=dict(zip(range(new_f.shape[0]),new_f.index))
        print("Data ingested in %s seconds." % (time.time()-start))
        return new_f.replace(dict_list)
    def predict(self,film_t,n):
        start=time.time()
        df=self.train(n)
        list_with_films = ["" for x in range(n)]
        message,item_id = self.find_film(film_t)
        if item_id == -1:
            return 'Please type more well-known film. Nothing was found'
        df_item=df[df.index==item_id]
        j=0
        for i in range(0,n):
            try:
                list_with_films[j]=self.movies['title'][(self.movies['movieId']==df_item.values[0,i])].iloc[0]
            except:
                print ('Smth is wrong with %i recommendation' % (j+1))
                j +=1
                continue
            j +=1
        print('Recommendation is produced in {} seconds for film {}'.format(time.time()-start,self.movies['title'][(self.movies['movieId']==item_id)].iloc[0]))    
        print message
        return list_with_films
    def find_film(self,text):
        message = ''
        find_f = self.movies[self.movies['title'].str.upper().str.contains(text.upper())]
        if find_f.shape[0]==0 :
            message = 'Please type more well-known film. Nothing was found'
            self.film_f=-1
        elif find_f.shape[0]>1:
            self.film_f=find_f.iloc[0,0]
            message = 'Ambigious result.If result is not best fit for you ,please type something more specific.'
        else:
            message = 'All is ok.'
            self.film_f = find_f.iloc[0,0]
        return message,self.film_f