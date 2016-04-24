# lang-stats-sp2016

11-761 Project solution for Daniel Clothiaux, Joana Correia, Aman Gupta, Elliot Schumacher, Emily Tagtow.
Implemented in Python 2.7

Run by executing the following command;
>python RunMe.py < TestSet.dat > TestPredictions.dat

Dependences
--------
Python packages : sklearn, nltk, numpy, gensim (can be installed by pip install [package])

SRILM Package (http://www.speech.sri.com/projects/srilm/download.html).  There is an installation tutorial here (http://www1.icsi.berkeley.edu/~wooters/SRILM/index.html).  Please install this in our project directory under [ProjectDirectory]/ngram. If, depending on your machine architecture, please make note that if your folder structure of SRILM does not match "ngram/lm/bin/macosx-m64/ngram", specifically the macosx-m64 portion, please add the folder name (macosx-m32 or whatever) as an argument to the python RunMe.py file (for example, python RunMe.py "macosx-m32" < TestSet.dat > TestPredictions.dat).

Java Version 6 or higher 
