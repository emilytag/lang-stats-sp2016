#utils.py

def read_data(file_name):
	"""
	Convert training/test/dev data into a list of documents with extra sentence tags removed.
	"""
	file_contents = open(file_name, 'r').read()
	docs = file_contents.split('~~~~~')[1:]

	for index, doc in enumerate(docs):
		doc = doc.replace("<s>", "")
		doc = doc.replace(" </s> \n", ".")
		doc = doc.strip()
		docs[index] = doc
	return docs

def read_labels(label_file_name):
	"""
	"""
	file_contents = open(label_file_name, 'r').readlines()
	return  [int(label.strip()) for label in file_contents]