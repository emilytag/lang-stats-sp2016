'''
Created on Apr 15, 2016

@author: elliotschumacher
'''


# setup bllip
from __future__ import print_function

from _collections import defaultdict
# from bllipparser import RerankingParser
# from bllipparser.ModelFetcher import download_and_install_model
import operator
import pickle
import sys

def buildSubtrees(tree):
    levels = defaultdict(str)
    levels[0] = tree.label
    if (len(tree) > 0):
        levels[1] = tree.label + " ( "
        levels[2] = tree.label + " ( "
        for subtree1 in tree:
            levels[1] += subtree1.label + " "
            levels[2] += subtree1.label + " ( "
            for subtree2 in subtree1:
                levels[2] += subtree2.label + " "
            levels[2] += ") "
        levels[1] += ")"
        levels[2] += ")"
    return set([x.replace(" ( ) ", " ") for x in levels.values()])



# def features(docList):
#     import time
# 
#     
#     # download model (only needs to be done once)
#     model_dir = download_and_install_model('WSJ', '/tmp/models')
#     # Loading the model is slow, but only needs to be done once
#     rrp = RerankingParser.from_unified_model_dir(model_dir)
#     rrp.set_parser_options(nbest = 5)
#     features = []
#     scores = []
#     with open("output_log.txt", "w") as logF, open("syn_feats.pkl", "w")  as synFile, open("syn_scores.pkl", "w")  as scoresFile:
# 
#         for i, doc in enumerate(docList):
#             start_time = time.time()
# 
#             features.append(defaultdict(float))
#             scores.append(defaultdict(list))
# 
#             for sentence in doc:
#                 
#                 parses = rrp.parse(sentence, rerank=False)
#                 #print(len(parses))
#                 #print(sentence, file = logF)
#                 try:
#                     parse_score = parses[0].parser_score
#                     rerank_score = parses[0].reranker_score
#                     scores[i]['parse'].append(parse_score)
#                     scores[i]['rerank'].append(rerank_score)
#                     scores[i]['sent_length'].append(len(parses[0].ptb_parse.tokens()))
#     
#                     best_parse = parses[0].ptb_parse
#                     # print(best_parse, file = logF)
#                 
#                     for t in best_parse.all_subtrees():
#                         levels = buildSubtrees(t)
#                         for l in levels:
#                             features[i][l] += 1.0
#                 except:
#                     print("No parse available - skipping")
#             features[i] = {x:v for x,v in features[i].items()}
#             print("{0}".format(sorted(features[i].items(), key=operator.itemgetter(1), reverse=True)), file = logF)
#             print("--- {0} seconds for {1} sentences ---" .format(time.time() - start_time, len(doc)))
# 
#         pickle.dump(features, synFile)
#         pickle.dump(scores, scoresFile)


#     t_bllip = Timer(lambda: rrp.parse(sentence))
#     print ("bllip", t_bllip.timeit(number=5))
    
    pass


def readfile(docFilename):
    docs = []
    count = 0
    with open(docFilename) as docfile, open('trainingSet.txt', 'w') as trainingF:
        
        for line in docfile:
            line = line.strip()
            if (line == "~~~~~"):
                docs.append([])
                trainingF.write("ENDOFDOC . \n")
            else:
                docs[-1].append( line.lower().replace("<s>", "").replace("</s>", "").strip())
                trainingF.write(line.lower().replace("<s>", "").replace("</s>", "").strip() + " . \n")
                count += 1
    print(count)
    return docs
if __name__ == '__main__':
    docs =  readfile('trainingSet.dat')

