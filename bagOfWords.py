import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import os
import pickle

def load(file_name):
	feats = []
	fname = "bow.pkl"
	if not os.path.isfile(fname) :
		file_contents = open(file_name, 'r').read()
		docs = file_contents.split('~~~~~')[1:]
	
		for index, doc in enumerate(docs):
			doc = doc.replace("<s>", "")
			doc = doc.replace(" </s> \n", ".")
			doc = doc.strip()
			docs[index] = doc
	
		count_vect = CountVectorizer()
		X_counts = count_vect.fit_transform(docs)
			
		## Create bag-of-words
		tf_transformer = TfidfTransformer(use_idf=False).fit(X_counts)
		X_tf = tf_transformer.transform(X_counts)
	
		## 
	
		for i, doc in enumerate(docs):
			row = X_tf.getrow(i)
			vals = {"BOW_"+str(i):row[0,i] for i in xrange(row.shape[1])}
			feats.append(vals)
		with open(fname, "wb") as pfile:
			pickle.dump(feats, pfile)
	else:
		with open(fname, "rb") as pfile:
			feats = pickle.load(pfile)


	return feats
