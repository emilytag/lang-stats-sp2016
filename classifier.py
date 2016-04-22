from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.feature_extraction import DictVectorizer
import sys
from syntax import trainSyntax, devtestSyntax
import subprocess
import re

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
      featureSet["uniquewords"] = len(set(article.split()))/len(article.split())
      featureSet.update(feats)
      temp_output_file = open('test_sent.txt', "w")
      temp_output_file.write(article.strip()+"\n")
      temp_output_file.close()
      command =  "ngram/lm/bin/macosx-m64/ngram -ppl " + 'test_sent.txt' + " -order 5 -lm ngram/LM-train-100MW.5.lm"
      output = subprocess.check_output(command, shell=True)
      #print article.strip()
      #print output.strip()
      ppl = re.search(r'ppl= \d*\.\d*', output)
      featureSet["ppl-5"] = float(ppl.group().split('=')[1])
      return featureSet

    def learn(self, article, classification, feats):
      features = self.extract_features(article, feats)
      self.allFeatures.append(features)
      self.allCorrect.append(classification)

    def fitModel(self):
      X = self.vec.fit_transform(self.allFeatures).toarray()
      y = np.array(self.allCorrect)
      self.model.fit(X, y)

    def predict(self, article, feats):
      features = self.extract_features(article, feats)
      f = self.vec.transform(features).toarray()
      prediction = self.model.predict(f)
      prob = self.model.predict_proba(f)
      return self.model.predict(f)[0]

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
    if pred == dev_labels[i]:
      correct_preds += 1
  print "model accuracy:", float(correct_preds)/len(dev_labels)

main()