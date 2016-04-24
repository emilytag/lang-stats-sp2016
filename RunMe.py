from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.feature_extraction import DictVectorizer
import sys
from syntax import trainSyntax, devtestSyntax
import subprocess
from sklearn.feature_selection.from_model import SelectFromModel

from sklearn.ensemble.forest import RandomForestClassifier
import pickle
import nltk
import itertools
from nltk.corpus import stopwords
import os
import re
from _collections import defaultdict
from datetime import datetime

foldername = "bow_syntax_pos_dumb"
class LogRegModel:
    def __init__(self,model = None, vec= None, featureselector= None):

        self.model = model
        self.vec = vec
        self.featSelect = featureselector
        


    def extract_features(self, article, feats, threegram_sent_ppl, fourgram_sent_ppl, fivegram_sent_ppl, sixgram_sent_ppl, index = None):
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

      try:
        sents = [x for x in article.split("\n") if len(x) > 1]
        ppl_five = ppl_wrangling(sents, fivegram_sent_ppl)
        ppl_six = ppl_wrangling(sents, sixgram_sent_ppl)
        ppl_three = ppl_wrangling(sents, threegram_sent_ppl)
        ppl_four = ppl_wrangling(sents, fourgram_sent_ppl)
        featureSet["ppl-5"] = ppl_five
        featureSet["ppl-6"] = ppl_six
        featureSet["ppl-3"] = ppl_three
        featureSet["ppl-4"] = ppl_four
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

              fs[pos + "_Run"] = max(fs[pos + "_Run"], max([sum(1 for _ in l) for n, l in itertools.groupby(postags) if n == pos]))
              
      fs = {x:v/totalCount for x,v in fs.items()}
      #fs["NNP_Perc"] = nnpCount / totalCount
      return fs
      
    def getPOSTags(self, article):
      articleSents = list(filter(bool, [line.lower().replace("<s>", "").replace("</s>", "").strip().split() for line in article.split("\n")]))
      postags = nltk.pos_tag_sents(articleSents)
      return postags
#       
#     def learn(self, article, classification, feats, index, threegram_sent_ppl, fourgram_sent_ppl, fivegram_sent_ppl, sixgram_sent_ppl):
#       features = self.extract_features(article, feats, threegram_sent_ppl, fourgram_sent_ppl, fivegram_sent_ppl, sixgram_sent_ppl, index)
#       self.allFeatures.append(features)
#       self.allCorrect.append(classification)
# 
#     def fitModel(self):
#       X = self.vec.fit_transform(self.allFeatures).toarray().astype(np.float)
#       y = np.array(self.allCorrect).astype(np.float)
#       self.featSelect = SelectFromModel(RandomForestClassifier()).fit(X,y)
#       #lr = 
#       #self.featSelect = RandomizedLogisticRegression().fit(X,y)#SelectFromModel(lr,prefit=True
# 
#       usePreloaded = False
#       featSelectFilename = os.path.join(foldername,"featselect_{0}.pkl".format(self.currTimestamp))
#       vecFilename = os.path.join(foldername,"vec_{0}.pkl".format(self.currTimestamp))
#       modelFilename = os.path.join(foldername,"model_{0}.pkl".format(self.currTimestamp))
# 
#       if (not usePreloaded):
# 
#           with open(featSelectFilename, 'wb') as featSelectF, open(vecFilename, 'wb') as vecF:
#               pickle.dump(self.featSelect, featSelectF)
#               pickle.dump(self.vec, vecF)
#               
#       else:
#           with open("featselect_2016-04-23 14:07:16.366972.pkl", 'rb') as featselectF:
#               self.featSelect = pickle.load(featselectF)
#       X = self.featSelect.transform(X)
#       print(X.shape)
#       self.printSelectedFeats()
#       self.model.fit(X, y)
#       with open(modelFilename, 'wb') as modelF:
#           pickle.dump(self.model, modelF)
#       #self.model.fit(X,y)
# 
#     def printSelectedFeats(self):
#       featIndxs = [i for i,x in enumerate(self.featSelect.get_support()) if x == True]
#       featlist = []
#       for feat, indx in self.vec.vocabulary_.items():
#         if indx in featIndxs:
#           featlist.append(feat)
#       for feat in sorted(featlist):
#         print("Selected feature:{0}".format(feat))
#       with open(os.path.join(foldername,"picked_feats_{0}.pkl".format(self.currTimestamp)), "wb") as picklefile:
#         pickle.dump(sorted(featlist), picklefile, protocol=2)
#         
#     def printCoefs(self):
#       featIndxs = [i for i,x in enumerate(self.featSelect.get_support()) if x == True]
#       featlist = []
#       coefs= self.model.coef_
#       for feat, indx in self.vec.vocabulary_.items():
#         if indx in featIndxs:
#           featlist.append((feat, coefs[0][len(featlist)]))
#       with open(os.path.join(foldername,"featurecoeffs_{0}.txt".format(self.currTimestamp)), "wb") as picklefile:
#        for feat, co in sorted(featlist, key=lambda x: x[1]):
#            picklefile.write("{0}:{1}\n".format(feat, co))

    def predict(self, article, feats, threegram_sent_ppl, fourgram_sent_ppl, fivegram_sent_ppl, sixgram_sent_ppl):
      features = self.extract_features(article, feats, threegram_sent_ppl, fourgram_sent_ppl, fivegram_sent_ppl, sixgram_sent_ppl)
      f = self.vec.transform(features).toarray()
      #prediction = self.model.predict(f)
      #prob = self.model.predict_proba(f)
      f = self.featSelect.transform(f)
      softprobs = self.model.predict_proba(f)
      predlabel = int(self.model.predict(f)[0])
      ind0 = np.where(np.float(0) == self.model.classes_)
      ind1 = np.where(np.float(1) == self.model.classes_)
      prob0 = float(softprobs[0][ind0])
      prob1 = float(softprobs[0][ind1])
      return predlabel , prob0, prob1

def ppl_wrangling(sents, sent_ppl):
  logprob_total = 0.0
  words_total = 0.0
  oovs_total = 0.0
  sents_total = 0.0
  for sent in sents:
    #print sent
    sents_total += 1
    for ppl in sent_ppl:
      #print ppl.split("\n")[0]
      if ppl.split("\n")[0] == sent.strip():
        logprob = re.search(r'logprob= -?\d*\.?\d*', ppl.split("\n")[2])
        logprob_total += float(logprob.group().split('=')[1])
        words = re.search(r'\d* words', ppl.split("\n")[1])
        words_total += float(words.group().split()[0])
        oovs = re.search(r'\d* OOVs', ppl.split("\n")[1])
        oovs_total += float(oovs.group().split()[0])
        break
  doc_ppl = 10.0 ** (-logprob_total/(words_total-oovs_total+sents_total))
  return doc_ppl

def ngram_ppls(filename):
  command3gram =  "ngram/lm/bin/macosx-m64/ngram -ppl " + filename + " -order 3 -lm ngram/LM-train-100MW.3grambin.lm -debug 1"
  output3gram = subprocess.check_output(command3gram, shell=True)
  threegram_sent_ppl = output3gram.split("\n\n")
  command4gram =  "ngram/lm/bin/macosx-m64/ngram -ppl " + filename + " -order 4 -lm ngram/LM-train-100MW.4grambin.lm -debug 1"
  output4gram = subprocess.check_output(command4gram, shell=True)
  fourgram_sent_ppl = output4gram.split("\n\n")
  command5gram =  "ngram/lm/bin/macosx-m64/ngram -ppl " + filename + " -order 5 -lm ngram/LM-train-100MW.5grambin.lm -debug 1"
  output5gram = subprocess.check_output(command5gram, shell=True)
  fivegram_sent_ppl = output5gram.split("\n\n")
  command6gram =  "ngram/lm/bin/macosx-m64/ngram -ppl " + filename + " -order 6 -lm ngram/LM-train-100MW.6grambin.lm -debug 1"
  output6gram = subprocess.check_output(command6gram, shell=True)
  sixgram_sent_ppl = output6gram.split("\n\n")
  return threegram_sent_ppl, fourgram_sent_ppl, fivegram_sent_ppl, sixgram_sent_ppl

def main():
  start = datetime.now()

  model = LogRegModel()
  modelTimestamp = "2016-04-24 13:13:00.547227"
  featSelectFilename = os.path.join(foldername,"featselect_{0}.pkl".format(modelTimestamp))
  vecFilename = os.path.join(foldername,"vec_{0}.pkl".format(modelTimestamp))
  modelFilename = os.path.join(foldername,"model_{0}.pkl".format(modelTimestamp))
  with open(featSelectFilename, 'rb') as featF, open(vecFilename, 'rb') as vecF, open(modelFilename, 'rb') as modelF:
      vec = pickle.load(vecF)
      feat = pickle.load(featF)
      modelObj = pickle.load(modelF)
      model = LogRegModel(model=modelObj, featureselector=feat, vec=vec)
      model.currTimestamp = modelTimestamp
#       
#   else:
#       train_data = open('trainingSet.dat', 'r').read()
#       train_labels = open('trainingSetLabels.dat', 'r').readlines()
#       train_data = train_data.split('~~~~~')[1:]
#       ngram_file = open('ngram_file_train.txt', 'w')
#       train_labels = [x.strip() for x in train_labels]
# 
#       for article in train_data:
#         ngram_file.write(article)
#       threegram_sent_ppl = []
#       fourgram_sent_ppl =[]
#       fivegram_sent_ppl=[]
#       sixgram_sent_ppl = []  
#       threegram_sent_ppl, fourgram_sent_ppl, fivegram_sent_ppl, sixgram_sent_ppl = ngram_ppls('ngram_file_train.txt')
#       
#       trainSyntaxFeats = trainSyntax.load()
#       #trainBagOfWordsFeats = bagOfWords.load('trainingSet.dat')
#     
#       for i in range(0, len(train_data)):#len(train_data)):
#         #print "sent number", i, datetime.now() - start 
#         feats = trainSyntaxFeats[i]
#         #feats.update(trainBagOfWordsFeats[i])
#         #Can add more features to feats object if more precomputed features are added
#         model.learn(train_data[i], train_labels[i], feats, i, threegram_sent_ppl, fourgram_sent_ppl, fivegram_sent_ppl, sixgram_sent_ppl)
#         if (i % 50 == 0):
#             print("Articles processed:{0}".format(i))
#             pass
#       with open("pos_tags.pkl", "wb") as postagsfile:
#           pickle.dump(model.posTagsDict, postagsfile)
#       model.fitModel()

  dev_labels = []

  dev_filename = "testSet.dat"
  dev_data = sys.stdin.read()
  
  with open(dev_filename, 'w') as testfile:
      testfile.write(dev_data)
  dev_data = dev_data.split('~~~~~')[1:]

      
  ngram_file_devtest = open('ngram_file_devtest.txt', 'w')
  for article in dev_data:
    ngram_file_devtest.write(article)
  threegram_sent_ppl = []
  fourgram_sent_ppl =[]
  fivegram_sent_ppl=[]
  sixgram_sent_ppl = []  
  #threegram_sent_ppl, fourgram_sent_ppl, fivegram_sent_ppl, sixgram_sent_ppl = ngram_ppls('ngram_file_devtest.txt')

  dev_labels = [x.strip() for x in dev_labels]
  correct_preds = 0
  
  devSyntaxFeats = devtestSyntax.generate(dev_filename)
  for i in range(0, len(dev_data)):
    feats = devSyntaxFeats[i]

    pred, prob0, prob1 = model.predict(dev_data[i], feats, threegram_sent_ppl, fourgram_sent_ppl, fivegram_sent_ppl, sixgram_sent_ppl)
    print("{0} {1} {2}".format(prob0, prob1, pred))
    if (len(dev_labels) > 0):
        if pred == int(dev_labels[i]):
          correct_preds += 1
  if (len(dev_labels) > 0):  
      with open(os.path.join(foldername,"results.txt"), "a+") as resfile:
          resfile.write("Accuracy:{0},FeatSet:{1}\n".format( float(correct_preds)/len(dev_labels), model.currTimestamp))       


main()
