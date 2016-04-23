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
import itertools
from nltk.corpus import stopwords
import os
import re
from _collections import defaultdict

class LogRegModel:
    def __init__(self):
      self.model = LogisticRegression()
      self.vec = DictVectorizer()
      self.allFeatures = []
      self.allCorrect = []
      self.posTagsDict = defaultdict(list)
      if (os.path.isfile("pos_tags.pkl") ):
          with open("pos_tags.pkl", "rb") as posfile:
              self.posTagsDict = pickle.load(posfile)


    def extract_features(self, article, feats, index = None):
      featureSet = {}
      
      articleWords = article.replace("<s>", "").replace("</s>", "").split()
      featureSet["articlelen"] = len(articleWords)
      fx_words = [word for word in articleWords if word.lower() in stopwords.words('english')]
      featureSet["fxwordcount"] = len(fx_words)/len(articleWords)
      non_words = [word for word in articleWords if word.isalpha() != True]
      featureSet["nonwordcount"] = len(non_words)/len(articleWords)
      content_words = [word for word in articleWords if word.lower() not in stopwords.words('english')]
      featureSet["contentwordcount"] = len(content_words)/len(articleWords)
      featureSet["uniquewords"] = len(set(articleWords))/len(articleWords)
      featureSet.update(feats)
      temp_output_file = open('test_sent.txt', "w")
      temp_output_file.write(article.strip()+"\n")
      temp_output_file.close()
      try:
          command6gram =  "ngram/lm/bin/macosx-m64/ngram -ppl " + 'test_sent.txt' + " -order 6 -lm ngram/LM-train-100MW.6gram.lm"
          output6gram = subprocess.check_output(command6gram, shell=True)
          ppl6gram = re.search(r'ppl= \d*\.?\d*', output6gram)
          featureSet["ppl-6"] = float(ppl6gram.group().split('=')[1])
      except:
          pass
      try:
          command5gram =  "ngram/lm/bin/macosx-m64/ngram -ppl " + 'test_sent.txt' + " -order 5 -lm ngram/LM-train-100MW.5gram.lm"
          output5gram = subprocess.check_output(command5gram, shell=True)
          ppl5gram = re.search(r'ppl= \d*\.?\d*', output5gram)
          featureSet["ppl-5"] = float(ppl5gram.group().split('=')[1])
      except:
          pass
      featureSet.update(self.posTags(index, article))
      return featureSet

    def posTags(self, index, article):
      fs = defaultdict(float)
      nnpCount = 0.0
      totalCount = 0.0
      ptd = []
      if (index is None):
          ptd = self.getPOSTags(article)
      else:
          if index not in self.posTagsDict:
              self.posTagsDict[index] = self.getPOSTags(article)
          ptd = self.posTagsDict[index]
      for posSent in ptd:
          postags = [x[1] for x in posSent]
          postagsset = set(postags)
          totalCount += float(len(postags))
          for pos in postagsset:
              fs[pos + "_Percent"] += sum(1.0 for x in postags if x == pos)

              fs[pos + "_Run"] = max(fs[pos + "_Run"], max([sum(1 for _ in l) for n, l in itertools.groupby(postags) if n == pos], default=0))
              
      fs = {x:v/totalCount for x,v in fs.items()}
      #fs["NNP_Perc"] = nnpCount / totalCount
      return fs
      
    def getPOSTags(self, article):
      articleSents = list(filter(bool, [line.lower().replace("<s>", "").replace("</s>", "").strip().split() for line in article.split("\n")]))
      postags = nltk.pos_tag_sents(articleSents)
      return postags
      
    def learn(self, article, classification, feats, index):
      features = self.extract_features(article, feats, index)
      self.allFeatures.append(features)
      self.allCorrect.append(classification)

    def fitModel(self):
      X = self.vec.fit_transform(self.allFeatures).toarray().astype(np.float)
      y = np.array(self.allCorrect).astype(np.float)
      self.featSelect = SelectFromModel(RandomForestClassifier()).fit(X,y)
      #lr = 
      #self.featSelect = RandomizedLogisticRegression().fit(X,y)#SelectFromModel(lr,prefit=True

      usePreloaded = False
      if (not usePreloaded):
          featSelectFilename = "featselect_{0}.pkl".format(datetime.datetime.now())
          with open(featSelectFilename, 'wb') as featSelectF:
              pickle.dump(self.featSelect, featSelectF)
      else:
          with open("featselect_2016-04-23 11:00:37.443495.pkl", 'rb') as featselectF:
              self.featSelect = pickle.load(featselectF)
      X = self.featSelect.transform(X)
      print(X.shape)
      self.printSelectedFeats()
      self.model.fit(X, y)
      #self.model.fit(X,y)

    def printSelectedFeats(self):
      featIndxs = [i for i,x in enumerate(self.featSelect.get_support()) if x == True]
      featlist = []
      for feat, indx in self.vec.vocabulary_.items():
        if indx in featIndxs:
          featlist.append(feat)
      for feat in sorted(featlist):
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
    model.learn(train_data[i], train_labels[i], feats, i)
  if (not os.path.isfile("pos_tags.pkl") ):
      with open("pos_tags.pkl", "wb") as postagsfile:
          pickle.dump(model.posTagsDict, postagsfile)
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