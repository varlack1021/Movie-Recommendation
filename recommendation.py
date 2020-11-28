import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import warnings; warnings.simplefilter('ignore')

from scipy import stats
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from pprint import pprint
from nltk.stem.snowball import SnowballStemmer

#---------------Import Data ------
df = pd.read_csv('archive/movies_metadata.csv')
links_small = pd.read_csv('archive/links_small.csv')
credits = pd.read_csv('archive/credits.csv')
keywords = pd.read_csv('archive/keywords.csv')

#Smaller subset, since I will e doing vector analysis using 9000 rows is better than 45000
links_small = links_small[links_small['tmdbId'].notnull()]['tmdbId'].astype('int')

#----------------Clean and Process Data---------
stemmer = SnowballStemmer('english')

credits = credits[credits['id'].notnull()]
credits['id'] = credits['id'].astype('int')

keywords = keywords[keywords['id'].notnull()]
keywords['id'] = keywords['id'].astype('int')

df['genres'] = df['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else x)
df['year'] = pd.to_datetime(df['release_date'], errors='coerce').apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)
df = df.drop([19730, 29503, 35587])
df['id'] = df['id'].astype('int')
df = df.merge(credits, on='id')
df = df.merge(keywords, on='id')

#smaller dataframe
sdf = df[df['id'].isin(links_small)]

sdf['tagline'] = sdf['tagline'].fillna('')
sdf['description'] = sdf['overview'] + sdf['tagline']
sdf['description'] = sdf['description'].fillna('')
sdf['crew'] = sdf['crew'].apply(literal_eval)
sdf['cast'] = sdf['cast'].apply(literal_eval)
sdf['keywords'] = sdf['keywords'].apply(literal_eval)

def get_director(movie):
	for crew_member in movie:
		if crew_member['job'] == 'Director':
			return crew_member['name']
	return np.nan

sdf['director'] = sdf['crew'].apply(get_director)
sdf['cast'] = sdf['cast'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
sdf['cast'] = sdf['cast'].apply(lambda x: x[:3] if len(x) >= 3 else x)

sdf['keywords'] = sdf['keywords'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
sdf['cast'] = sdf['cast'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x]) 
sdf['director'] = sdf['director'].astype('str').apply(lambda x: str.lower(x.replace(" ", "")))
sdf['director'] = sdf['director'].apply(lambda x: [x, x, x])

value_counts = sdf.apply(lambda x: pd.Series(x['keywords']), axis=1).stack().reset_index(level=1, drop=True)
value_counts.name = 'keyword'
value_counts = value_counts.value_counts()
value_counts = value_counts[value_counts > 1]

#We remove keywords that only occur once
def filter_keywords(keywords):
	words = []
	for word in keywords:
		if word in value_counts:
			words.append(word)
	return words

sdf['keywords'] = sdf['keywords'].apply(filter_keywords)
sdf['keywords'] = sdf['keywords'].apply(lambda x: [stemmer.stem(word) for word in x])
sdf['keywords'] = sdf['keywords'].apply(lambda x: [str.lower(word.replace(" ", "")) for word in x])

#data dump
sdf['dataDump'] = sdf['genres'] + sdf['director'] + sdf['keywords'] + sdf['cast']
sdf['dataDump'] = sdf['dataDump'].apply(lambda x: " ".join(x))

sdf = sdf.reset_index()

#-------Rating Algorithms------------

#not null returns a list of booleans
#passing the list of booleans tells us which rows to include
def get_top_movies():
	global df
	vote_counts = df[df['vote_count'].notnull()]['vote_count'].astype('int')
	vote_averages = df[df['vote_average'].notnull()]['vote_average'].astype('int')
	C = vote_averages.mean()
	m = vote_counts.quantile(.99)

	#demonstrates strong bias
	qualified = df[(df['vote_count'] >= m) & (df['vote_count'].notna()) & (df['vote_average'].notna())] 
	qualified = qualified[['title', 'year', 'vote_count', 'vote_average', 'popularity', 'genres']]


	def movie_rating(movie):
		v = movie['vote_count']
		R = movie['vote_average']
		return str((v/(v+m) * R) + (m/(m+v) * C))
	
	qualified['weighted_rating'] = qualified.apply(movie_rating, axis=1)
	qualified = qualified.sort_values('weighted_rating', ascending=False)
	return qualified['title']

#---------By genre------
def top_movie_by_genre(genre, percentile=.85):
	#Make a column to easily search movies by genre
	global df
	genres = df.apply(lambda x: pd.Series(x['genres']), axis=1).stack().reset_index(level=1, drop=True)
	genres.name = 'genres'
	genres_df = df.drop('genres', axis=1).join(genres)
	
	df = genres_df[genres_df['genres'] == genre]
	vote_counts = df[df['vote_count'].notnull()]['vote_count'].astype('int')
	vote_averages = df[df['vote_average'].notnull()]['vote_average'].astype('int')
	C = vote_averages.mean()
	m = vote_counts.quantile(percentile)

	qualified = df[(df['vote_count'] >= m) & (df['vote_count'].notnull()) & (df['vote_average'].notnull())]
	qualified = qualified[['title', 'year', 'vote_count', 'vote_average', 'popularity']]

	qualified['weighted_rating'] = qualified.apply(lambda x : (x['vote_count']/(x['vote_count']+m) * x['vote_average']) + (m/(m+x['vote_count']) * C), axis=1)
	qualified = qualified.sort_values('weighted_rating', ascending=False).head(250)
	return qualified

def get_recommedation_from_similarities(title, cosine_sim, titles, indices):
	idx = indices[title]
	sim_scores = list(enumerate(cosine_sim[idx]))
	sim_scores = sorted(sim_scores, key=lambda x : x[1], reverse=True)
	sim_scores =  sim_scores[1:31]
	movie_indices = [i[0] for i in sim_scores]
	return {titles.iloc[i[0]]: i[1] for i in sim_scores}

def get_recommendations_with_ratings(title, percentile, indices):
	#Organizes by ratings first and then filters by cosine_similarity
	##idx = indices[title]
	'''
	sim_scores = list(enumerate(cosine_sim[idx]))
	sim_scores = sorted(sim_scores, key=lambda x : x[1], reverse=True)
	sim_scores = sim_scores[1:250]	
	movie_indices = [i[0] for i in sim_scores]
	similar_movies = sdf.iloc[movie_indices]
	'''
	vote_counts = sdf[sdf['vote_count'].notnull()]['vote_count'].astype('int')
	vote_averages = sdf[sdf['vote_average'].notnull()]['vote_average'].astype('int')
	C = vote_averages.mean()
	m = vote_counts.quantile(percentile)

	qualified = sdf[(sdf['vote_count'] >= m) & (sdf['vote_count'].notnull()) & (sdf['vote_average'].notnull())]
	qualified = qualified[['title', 'year', 'vote_count', 'vote_average', 'popularity']]

	qualified['weighted_rating'] = qualified.apply(lambda x : (x['vote_count']/(x['vote_count']+m) * x['vote_average']) + (m/(m+x['vote_count']) * C), axis=1)
	qualified = qualified.sort_values('weighted_rating', ascending=False).head(250)
	
	indices = pd.Series(qualified.index, index=qualified['title'])
	idx = indices[title]
	sim_scores = list(enumerate(cosine_sim[idx]))
	sim_scores = sorted(sim_scores, key= lambda x: x[1], reverse=True)
	sim_scores = sim_scores[1:31]
	movie_indices = [i[0] for i in sim_scores]
	return sdf.iloc[movie_indices]

'''
Example
lis = ['The sky is blue', 'The sun is bright', 	]
test_matrix = tf.fit_transform(lis)
for i, x  in test_matrix.items():
	print(i, x)
pprint(test_matrix.todense())
names = tf.get_feature_names()
test_cosim_sim = cosine_similarity(test_matrix, test_matrix)

test_df = pd.DataFrame(data=test_matrix.toarray(), columns=names, index=lis)
print(test_df)
pprint(test_cosim_sim)
'''

def main():
	#running this with the bigger dataset resulted in my computer being unable to handle it 15.4GB
	tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')
	tfidf_matrix = tf.fit_transform(sdf['description'])
	tfidf_cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

	idf_titles = sdf['title']
	tfidf_indices = pd.Series(sdf.index, index=sdf['title'])
	
	count = CountVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')
	count_matrix = count.fit_transform(sdf['dataDump'])
	count_cosine_sim = cosine_similarity(count_matrix, count_matrix)

	count_titles = sdf['title']
	count_indices = pd.Series(sdf.index, index=sdf['title'])

	result = get_recommedation_from_similarities('The Dark Knight', tfidf_cosine_sim, idf_titles, tfidf_indices)
	result2 = get_recommedation_from_similarities('The Dark Knight', count_cosine_sim, count_titles, count_indices)

	result = sorted(result.items(), key=lambda x: x[1], reverse=True)
	result2 = sorted(result2.items(), key=lambda x: x[1], reverse=True)

	'''
	pprint(get_top_movies())

	pprint(top_movie_by_genre('Romance'))

	pprint(result)
	print("----")
	pprint(result2)
	'''
main()