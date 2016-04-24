'''
Created on Apr 19, 2016

@author: elliotschumacher
'''
from __future__ import print_function

from nltk import Tree
from _collections import defaultdict
import pickle
import re
import operator
import subprocess
import os.path



def buildSubtrees(tree):
    levels = defaultdict(str)
    levels[0] = tree.label()
    if (len(tree) > 0):
        levels[1] = tree.label() + " ( "
        levels[2] = tree.label() + " ( "
        levels[3] = tree.label() + " ( "
        for subtree1 in tree:
            if (not isinstance(subtree1, str)):
                levels[1] += subtree1.label() + " "
                levels[2] += subtree1.label() + " ( "
                levels[3] += subtree1.label() + " ( "
                for subtree2 in subtree1:
                    if (not isinstance(subtree2, str)):
    
                        levels[2] += subtree2.label() + " "
                        levels[3] += subtree2.label() + " ( "
                        for subtree3 in subtree2:
                            if (not isinstance(subtree3, str)):
                                levels[3] += subtree3.label() + " "
                        levels[3] += ") "
                levels[2] += ") "
                levels[3] += ") "
        levels[1] += ") "
        levels[2] += ") "
        levels[3] += ") "
    levels.pop(3, None)
    return set([re.sub( '\s+', ' ', x.replace("( )", "") ).strip() for x in levels.values()])
def readfile(docFilename):
    docs = []
    count = 0
    procFilename = "syntax/" + docFilename.replace(".dat", ".txt")
    with open(docFilename) as docfile, open(procFilename, 'w') as trainingF:
        
        for line in docfile:
            line = line.strip()
            if (line == "~~~~~"):
                docs.append([])
                trainingF.write("ENDOFDOC . \n")
            else:
                docs[-1].append( line.lower().replace("<s>", "").replace("</s>", "").strip())
                trainingF.write(line.lower().replace("<s>", "").replace("</s>", "").strip() + " . \n")
                count += 1
    #print(count)
    #return docs
    return procFilename


def generate(datFilename, useCached = True):
    procFilename = readfile(datFilename)
    features = []
    scores = []

    inputfile = procFilename.replace(".txt", "parse.txt")
    procFilenamesub = procFilename.replace("syntax/", "")
    synfeatsfilename = "syntax/syn_feats_{0}.pkl".format(procFilenamesub.replace(".txt", ""))
    synScoreFilename = "syntax/syn_scores_{0}.pkl".format(procFilenamesub.replace(".txt", ""))
    if not useCached or not os.path.isfile(synfeatsfilename):

        with open(inputfile, 'w') as inputF:
            proc = subprocess.Popen(["java", "-cp", "syntax/stanford-parser-full-2014-10-31/stanford-parser.jar", "edu.stanford.nlp.parser.lexparser.LexicalizedParser", "syntax/stanford-parser-full-2014-10-31/englishPCFG.ser.gz", procFilename], stdout=inputF)
            proc.wait()
        with open("syntax/output_log_{0}".format(procFilenamesub), "w") as logF, open(synfeatsfilename, "wb")  as synFile, open(synScoreFilename, "wb")  as scoresFile, open(inputfile) as zparFile:
        
            lstr = zparFile.read()
            docs = lstr.split("\n\n")
            for l in docs:
                if ("(NP (NN ENDOFDOC) (. .)))" in l):
                    if (len(features) > 0):
                        features[-1] = {x:v for x,v in features[-1].items()}
                        print("{0}".format(sorted(features[-1].items(), key=operator.itemgetter(1), reverse=True)), file = logF)
                    features.append(defaultdict(float))
                    scores.append(defaultdict(list))
                elif (len(l.strip()) > 0):
    
                    tr= Tree.fromstring(l)[0]
        
    
        
        #             try:
        
                    scores[-1]['sent_length'].append(len(tr.leaves()))
                    # print(best_parse, file = logF)
                    for t in tr.subtrees():
                        levels = buildSubtrees(t)
                        for l in levels:
                            features[-1][l] += 1.0
            #             except:
            #                 print("No parse available - skipping")
    
                    
    
            pickle.dump(features, synFile)
            pickle.dump(scores, scoresFile)
    else:
        with open(synfeatsfilename, "rb")  as synFile, open(synScoreFilename, "rb")  as scoresFile:
            features = pickle.load(synFile)
            scores = pickle.load(scoresFile)
            
    feats = []    
    for i in range(len(features)):
        feats.append({"syntax_" +x:v/len(scores[i]['sent_length']) for x,v in features[i].items()})
    return feats



if __name__ == '__main__':
    generate()