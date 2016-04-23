from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.feature_extraction import DictVectorizer
import sys
from syntax import trainSyntax, devtestSyntax
import subprocess
from sklearn.feature_selection.from_model import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.linear_model.randomized_l1 import RandomizedLogisticRegression
from sklearn.ensemble.forest import RandomForestClassifier
import datetime
import pickle
import nltk

class LogRegModel:
    def __init__(self):
      self.model = LogisticRegression()
      self.vec = DictVectorizer()
      self.allFeatures = []
      self.allCorrect = []


    def extract_features(self, article, feats):
      featureSet = {}
      featureSet['article_len'] = len(article.split())
      fx_words = [word for word in article.split() if len(word) <= 4]
      featureSet["fxwordcount"] = len(fx_words)
      non_words = [word for word in article.split() if word.isalpha() != True]
      featureSet["nonwordcount"] = len(non_words)
      content_words = [word for word in article.split() if len(word) > 4]
      featureSet["contentwordcount"] = len(content_words) 
      #FEATURES TO DO
      featureSet["uniquewords"] = len(set(article.split()))/len(article.split())
      featureSet.update(feats)
      #temp_output_file = open('test_sent.txt', "w")
      #temp_output_file.write(article)
      #subprocess.call("ngram/ngram -ppl " + 'test_sent.txt' + " -order 4 -lm ngram/data_train.5.lm"
      text = word_tokenize("And now for something completely different")
      nltk.pos_tag(text)
      return featureSet

    def learn(self, article, classification, feats):
      features = self.extract_features(article, feats)
      self.allFeatures.append(features)
      self.allCorrect.append(classification)

    def fitModel(self):
      X = self.vec.fit_transform(self.allFeatures).toarray().astype(np.float)
      y = np.array(self.allCorrect).astype(np.float)
      self.featSelect = SelectFromModel(RandomForestClassifier()).fit(X,y)
      #lr = 
      #self.featSelect = RandomizedLogisticRegression().fit(X,y)#SelectFromModel(lr,prefit=True
      X = self.featSelect.transform(X)
      print(X.shape)
      self.printSelectedFeats()
      usePreloaded = True
      if (usePreloaded):
          featSelectFilename = "featselect_{0}.pkl".format(datetime.datetime.now())
          with open(featSelectFilename, 'wb') as featSelectF:
              pickle.dump(self.featSelect, featSelectF)
      else:
          with open("featselect_2016-04-22 15:25:06.304474.pkl", 'rb') as featselectF:
              self.featSelect = pickle.load(featselectF)
      self.model.fit(X, y)
      #self.model.fit(X,y)

    def printSelectedFeats(self):
      featIndxs = [i for i,x in enumerate(self.featSelect.get_support()) if x == True]
      for feat, indx in self.vec.vocabulary_.items():
        if indx in featIndxs:
          print("Selected feature:{0}".format(feat))
    def predict(self, article, feats):
      features = self.extract_features(article, feats)
      f = self.vec.transform(features).toarray()
      #prediction = self.model.predict(f)
      #prob = self.model.predict_proba(f)
      f = self.featSelect.transform(f)
      return int(self.model.predict(f)[0])

def main():
  train_data = open('trainingSet.dat', 'r').read()
  train_labels = open('trainingSetLabels.dat', 'r').readlines()
  train_data = train_data.split('~~~~~')[1:]
  train_labels = [x.strip() for x in train_labels]
  model = LogRegModel()
  trainSyntaxFeats = trainSyntax.load()

  for i in range(0, len(train_data)):
    feats = trainSyntaxFeats[i]
    #Can add more features to feats object if more precomputed features are added
    model.learn(train_data[i], train_labels[i], feats)

  model.fitModel()
  dev_filename = 'developmentSet.dat'
  dev_data = open(dev_filename, 'r').read()
  dev_labels = open('developmentSetLabels.dat', 'r').readlines()
  dev_data = dev_data.split('~~~~~')[1:]
  dev_labels = [x.strip() for x in dev_labels]
  correct_preds = 0
  
  devSyntaxFeats = devtestSyntax.generate(dev_filename)
  for i in range(0, len(dev_data)):
    feats = devSyntaxFeats[i]

    pred = model.predict(dev_data[i], feats)
    if pred == int(dev_labels[i]):
      correct_preds += 1
  print ("model accuracy:", float(correct_preds)/len(dev_labels))

main()
#trainSyntax.generate()