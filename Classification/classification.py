import pandas as pd
import numpy as np
from nltk.tokenize.regexp import RegexpTokenizer
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv("dataset.tab", sep='\t', encoding = "utf-8", names=['title', 'content', 'category', 'newsType'])
df = df.drop(0, axis=0)

# print(df.head())
df['combined'] = df['title'] + df['content']

# Inializing type of tokenizer, stemmer and stopwords
tokenizer = RegexpTokenizer(r'[a-zA-Z\']+')
stemmer = SnowballStemmer('english')
stopwords = stopwords.words('english')
with open('StopWords.txt', 'r') as file: 
    for i in file:
        stopwords.append(i.strip())


# Tokenize word
df['tokenized_words'] = df.apply(lambda row: tokenizer.tokenize((row['combined'])), axis=1)
df['tokenized_words'] = df['tokenized_words'].apply(lambda x: [item for item in x if item not in stopwords])

# Stemming words with Snowball Stemmer
df['tokenized_words'] = df['tokenized_words'].apply(lambda x: [stemmer.stem(word) for word in x])

# Lower case all the tokenized words
df = df.astype(str)
df['tokenized_words'] = df['tokenized_words'].str.lower()

X = df.loc[:, 'tokenized_words']
y = df['category']

tfidf = TfidfVectorizer()
X = tfidf.fit_transform(X)

'''
# Training model using cross-validation for 5 selected algorithm for comparision

from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier

# Random Forest 
rf = RandomForestClassifier(n_estimators=100, random_state=0)
scores = cross_val_score(rf, X, y, cv=10)
print("Random Forest Classifier score: {:.2f}".format(scores.mean()))

# K-Nearest Neighbor
knn = KNeighborsClassifier(n_neighbors=10)
scores = cross_val_score(knn, X, y, cv=10, )
print("KNN score: {:.2f}".format(scores.mean()))

# Naive Bayes
nb = MultinomialNB(alpha=0.1)
scores = cross_val_score(nb, X, y, cv=10)
print("NB score: {:.2f}".format(scores.mean()))

# Logistic Regression
lg = LogisticRegression(penalty='l2', solver='liblinear', C=10.0)
scores = cross_val_score(lg, X, y, cv=10)
print("Logistic Regression score: {:.2f}".format(scores.mean()))

# Support Vector Machine (Linear kernel)
svm = SVC(kernel='linear')
scores = cross_val_score(svm, X, y, cv=10)
print("SVM score: {:.2f}".format(scores.mean()))
'''

'''
# Training model using train test split method
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print("Shape of features:{}".format(X.shape))
print("Shape of target:{}".format(y.shape))

clf_dict = {
    "lg": LogisticRegression(penalty='l2', solver='liblinear', C=10.0),
    "svm": SVC(kernel='linear'),
    "nb": MultinomialNB(alpha=0.1),
    "knn": KNeighborsClassifier(n_neighbors=10),
    "rf": RandomForestClassifier(n_estimators=100, random_state=0)
}

clf = clf_dict['knn']
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
'''