import numpy as np
import time as time
# Import the pandas package, then use the "read_csv" function to read the labeled training data
import pandas as pd		# Allows to import the data
from bs4 import BeautifulSoup 	# Will be used to remove the HTML characters
import re			# This packages allows to remove punctuation and numbers
import nltk			# Allows to remove the stopwords (those words that carry not meaning, like 'and', 'the'...)
#nltk.download('all')		# Downloads the stopwords data sets
#print stopwords.words("english")
from nltk.corpus import stopwords	# Import a stopword list in English
from nltk.corpus import words

from sklearn.feature_extraction.text import CountVectorizer		# Allows us to use bag-of-words learning and vectorize the set
#from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn import cross_validation

# This is a problem of regression, therefore we need to use a regression package from scikit learn, and we are going to use a linear model at first.

from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from math import sqrt


# For some reason python3 does not understand sklearn, Python2 does.
#Also, Python2 have troubles with nltk. It may be due to an update of scipy that was not compatible with nltk anymore. I am now updating scipy and that solved the problem.


# Imports the data. The target is the sentiment
def train_data_import(data = "train.tsv"):
	return pd.read_csv(data, header=0, delimiter="\t", quoting=3)

def test_data_import(data = "test.tsv"):
	return pd.read_csv(data, header=0, delimiter="\t", quoting=3)



# The following function takes in some text and outputs a cleaned text
def clean_review( raw_text ):		# Removes the HTML, punctuation, numbers, stopwords...
	rm_html = BeautifulSoup(raw_text).get_text()	# removes html
	letters_only = re.sub("[^a-zA-Z]",           	# The pattern to search for; ^ means NOT
                   		  " ",                   	# The pattern to replace it with
                          rm_html )              	# The text to search
	lower_case = letters_only.lower()	         	# Convert to lower case
	words = lower_case.split()          	     	# Split into words
	stops = stopwords.words("english")
	stops.append('ve')
	stops = set(stops)
#	english_words = words.words()[1:100]
	meaningful_words = [w for w in words if not w in stops]	# Remove stop words from "words"
	return ' '.join(meaningful_words)			# Joins the words back together separated by a space

# The following function iterates clean_review over all reviews in the set
def clean_all_reviews( raw_train_data, N_articles ):
	cleaned_reviews = []
	for i in xrange(N_articles):
		cleaned_reviews.append(clean_review(raw_train_data["name"][i]))
		if ( (i+1) % 1000 == 0 ):
			print(' -- Clean review # %d' % i)
	return cleaned_reviews


# Initialize the "CountVectorizer" object, which is scikit-learn's bag of words tool.
def Bag_of_Words(cleaned_reviews, n_features = 100):
	vectorizer = CountVectorizer(analyzer = "word",
                             tokenizer = None,    	# Allows to tokenize
                             preprocessor = None, 	# Allows to do some preprocessing
                             stop_words = None,   	# We could remove stopwords from here
                             max_features = n_features) 	# Chooses a given number of words, just a subset of the huge total number of words.
	# fit_transform() does two functions: First, it fits the model
	# and learns the vocabulary; second, it transforms our training data
	# into feature vectors. The input to fit_transform should be a list of 
	# strings.
	data_features = vectorizer.fit_transform(cleaned_reviews)
	data_features = data_features.toarray()
	return data_features, vectorizer


# Counts the words that appear in the reviews
def word_count():
	df_train = train_data_import()
	N_articles = 100#len(raw_train_data["review"][:])
	cleaned_reviews = clean_all_reviews(df_train, N_articles)
	train_data_features, vectorizer = Bag_of_Words(cleaned_reviews, N_articles)
	vocab = vectorizer.get_feature_names()
	dist = np.sum(train_data_features, axis=0)
	word_count = sorted(zip(dist,vocab),reverse = True)
	for count, tag in word_count:
	    print '{}: {}'.format(count, tag)


def Trainer():
	Tstart = time.time()
	print '- Import all training reviews'
	df_train = train_data_import()
	N_articles = 100000 #len(df_train["name"])
	print '- Start cleaning the training reviews'
	cleaned_reviews = clean_all_reviews(df_train, N_articles)
	print '- Creating the bag-of-words with %d articles' % N_articles
	train_data_features, vectorizer = Bag_of_Words(cleaned_reviews)
	print '- Trains a classifier'
	X_train, X_valid, Y_train, Y_valid = cross_validation.train_test_split(train_data_features, df_train["price"][:N_articles], train_size = 0.8, random_state=10)
#	N_train = int(0.8 * N_articles)
#	X_train = train_data_features[:N_train]
#	Y_train = df_train["price"][:N_train]
#	X_valid = train_data_features[N_train:N_articles]
#	Y_valid = df_train["price"][N_train:N_articles]
	clf = RandomForestRegressor(max_depth=2, random_state=0)
	forest = clf.fit( X_train, Y_train )
	Y_predicted = clf.predict(X_valid)
	rms = sqrt(mean_squared_error(Y_valid, Y_predicted))	#The Kaggle competition uses rms as the metric
#	score = clf.score(X_valid, Y_valid)
	Tend = time.time()-Tstart
	print '- Finished training in %f s, score = %f' % (Tend, rms)
#	return train_data_features, score


#df_train = train_data_import()

#print(len(df_train["category_name"]))

#word_count()

Trainer()

























