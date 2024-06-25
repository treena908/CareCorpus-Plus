from yellowbrick.text.tsne import tsne
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter

import pandas as pd
path='./data/'

# Load the data and create document vectors
corpus=pd.read_csv(path+'train_forum_com_undersample_class4_1000.csv')
label_list=Counter(corpus['label'])
print(label_list)

# tfidf = TfidfVectorizer()
#
# X = tfidf.fit_transform(corpus.strategy)
# y = corpus.label
#
# tsne(X, y)