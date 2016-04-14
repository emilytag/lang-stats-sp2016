# lang-stats-sp2016
TO - DO
------------------
-Take a look at Roni's toolkit
-Set up a classifier


FIRST MEETING IDEAS
------------------

1. Sample from the 100 million words corpus to create more training data
2. Train n-gram models on the 100 million word corpus to find "true" distributions. Then, calculate cross entropy of real and fake articles n-gram distributions and use the value as a feature
3. LSTM after samplinf from the 100 million word corpus
4. Word vectors as features
5. Capture richer word dependencies that n-grams can manage


AMAN's THOUGHTS
----------------
1. POS tagging might lead to different distributions of tags for real and fake articles (although I am not very confident about this).
2. Distribution of n-grams (n>3) might be different for real and fake articles
3. Since most of the articles in development are a few sentences long, we should think of sentence-level features. Syntactical parsing of some kind might leas to different results for real and fake articles.


RONI's TOOLKIT - http://www.speech.cs.cmu.edu/SLM/toolkit.html




SECOND MEETING IDEAS
----------------

1. Set up pipeline for classification and feature generation
2. use 90's tooliit to build higher order n-gram from the training data a compute the model's perplexity to the validation data. find a threshlold to decie whether it is human/machine generated text.
3. Possible features - train different language models (higher-order n-grams is one option), stop words, parse subtrees, part-of-speech, word vectors, term frequencies, ground clusters. 


Work division
Dan - Sampling, ground clusters
Elliot - Parse trees
Aman - part-of-speech
Joana - Roni's toolkit - higher order n-grams.
Emily - n-grams, classification script

