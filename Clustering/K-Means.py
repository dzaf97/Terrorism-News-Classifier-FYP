import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn import metrics
from yellowbrick.cluster import SilhouetteVisualizer
import numpy as np


# Load input data
data = []
with open('dataset.csv', 'r') as f:
    for line in f.readlines():
        data.append(line[:-1])

print("Dataset Loaded...\n")

# Tokenization and Stemming
stemmer = SnowballStemmer('english')
tokenizer = RegexpTokenizer(r'[a-zA-Z\']+')

def tokenize(text):
    return [stemmer.stem(word) for word in tokenizer.tokenize(text.lower())]

#Adding new stopwords
stopwords = nltk.corpus.stopwords.words('english')
with open('StopWords.txt', 'r') as file: 
    for i in file:
        stopwords.append(i.strip())

# Feature extraction using TF-IDF
tfidf = TfidfVectorizer(stop_words = stopwords, tokenizer = tokenize)
X = tfidf.fit_transform(data)
words = tfidf.get_feature_names()
print(words)
print("Number of features: " + str(len(words))) #check number of features
print("Vectorization Completed...\n")

'''
# Clustering using K-Means
print("k-Means with 5 cluster:\n")
kmeans_1 = KMeans(init='k-means++', n_clusters = 5, random_state = 42)
kmeans_1.fit(X)
common_words = kmeans_1.cluster_centers_.argsort()[:,-1:-21:-1]
for num, centroid in enumerate(common_words):
    print(str(num) + ' : ' + ', '.join(words[word] for word in centroid))
print('\n')


print("k-Means with 7 cluster:\n")
kmeans_2 = KMeans(init='k-means++', n_clusters = 7, random_state = 42)
kmeans_2.fit(X)
common_words = kmeans_2.cluster_centers_.argsort()[:,-1:-21:-1]
for num, centroid in enumerate(common_words):
    print(str(num) + ' : ' + ', '.join(words[word] for word in centroid))
print('\n')

print("k-Means with 10 cluster:\n")
kmeans_3 = KMeans(init='k-means++', n_clusters = 10, random_state = 42)
kmeans_3.fit(X)
common_words = kmeans_3.cluster_centers_.argsort()[:,-1:-21:-1]
for num, centroid in enumerate(common_words):
    print(str(num) + ' : ' + ', '.join(words[word] for word in centroid))
print('\n')


#Silhouette Coefficient

labels_1 = kmeans_1.labels_
print("Score for 5 Cluster: " , str(metrics.silhouette_score(X, labels_1, metric='euclidean')))

labels_2 = kmeans_2.labels_
print("Score for 7 Cluster: " , str(metrics.silhouette_score(X, labels_2, metric='euclidean')))

labels_3 = kmeans_3.labels_
print("Score for 10 Cluster: " , str(metrics.silhouette_score(X, labels_3, metric='euclidean')))
'''

'''
visualizer = SilhouetteVisualizer(kmeans_1)
visualizer.fit(X) # Fit the training data to the visualizer
visualizer.poof() # Draw/show/poof the data

visualizer = SilhouetteVisualizer(kmeans_2)
visualizer.fit(X) # Fit the training data to the visualizer
visualizer.poof() # Draw/show/poof the data

visualizer = SilhouetteVisualizer(kmeans_3)
visualizer.fit(X) # Fit the training data to the visualizer
visualizer.poof() # Draw/show/poof the data
'''
