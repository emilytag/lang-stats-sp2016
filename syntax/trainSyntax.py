'''
Created on Apr 19, 2016

@author: elliotschumacher
'''
from __future__ import print_function

from nltk import Tree
from _collections import defaultdict
import pickle
import operator

featureFilename = "syntax/syn_feats_{0}.pkl"
scoreFilename = "syntax/syn_scores_{0}.pkl"
ver = "train"


def buildSubtrees(tree):
    levels = defaultdict(str)
    levels[0] = tree.label()
    if (len(tree) > 0):
        levels[1] = tree.label() + " ( "
        levels[2] = tree.label() + " ( "
        for subtree1 in tree:
            if (not isinstance(subtree1, str)):
                levels[1] += subtree1.label() + " "
                levels[2] += subtree1.label() + " ( "
                for subtree2 in subtree1:
                    if (not isinstance(subtree2, str)):
    
                        levels[2] += subtree2.label() + " "
                levels[2] += ") "
        levels[1] += ")"
        levels[2] += ")"
    return set([x.replace("( )", "").strip() for x in levels.values()])



def generate():
    features = []
    scores = []
    with open("output_log_{0}.txt".format(ver), "w") as logF, open(featureFilename.format(ver), "w")  as synFile, open(scoreFilename.format(ver), "w")  as scoresFile, open('zpar.txt') as zparFile:
        for l in zparFile:
            if (l == "(NP (NNP ENDOFDOC))\n"):
                if (len(features) > 0):
                    features[-1] = {x:v for x,v in features[-1].items()}
                    print("{0}".format(sorted(features[-1].items(), key=operator.itemgetter(1), reverse=True)), file = logF)
                features.append(defaultdict(float))
                scores.append(defaultdict(list))
            else:
                tr= Tree.fromstring(l)
    

    
    #             try:
    
                scores[-1]['sent_length'].append(len(tr.leaves()))
                # print(best_parse, file = logF)
                #print(tr)
                for t in tr.subtrees():
                    levels = buildSubtrees(t)
                    for l in levels:
                        #print (l)
                        features[-1][l] += 1.0
    #             except:
    #                 print("No parse available - skipping")


        pickle.dump(features, synFile)
        pickle.dump(scores, scoresFile)
    
    pass

def load():
    feats = []
    with open(featureFilename.format(ver), 'rb')  as synFile, open(scoreFilename.format(ver), 'rb')  as scoresFile:
        features = pickle.load(synFile)
        scores = pickle.load(scoresFile)
        for i in range (len(features)):
            feats.append({x:v/len(scores[i]['sent_length']) for x,v in features[i].items()})

    return feats
if __name__ == '__main__':
    load()