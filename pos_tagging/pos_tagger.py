import nltk
from collections import Counter

class PosTagger(object):
	"""PosTagger: """
	def __init__(self):
		super(PosTagger, self).__init__()

	def tag(self, text):
		"""Tag a sequence of text using Part-of-Speech
		Arguments:
		text - data to be tagged in free text format
		"""
		tokenized_text = nltk.word_tokenize(text)
		tagged_tokenized_text = nltk.pos_tag(tokenized_text)
		return tagged_tokenized_text

	def tag_corpus(self, docs, labels):
		"""
		Calculate distribution of POS tags in real and fake documents
		"""
		cntr_true = Counter()
		cntr_fake = Counter()
		for index, doc in enumerate(docs):
			res = self.tag(doc)
   			tag_res = [elem[1] for elem in res]
   			if (labels[index] == 0):
   				cntr_fake.update(tag_res)
   			else:
   				cntr_true.update(tag_res)

   		return cntr_true, cntr_fake