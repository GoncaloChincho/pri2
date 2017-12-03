import nltk
import nltk.data
import re

from nltk.corpus import stopwords
from nltk.stem.porter import *
from wordsegment import load, segment

cachedStopWords = stopwords.words("english")


#Build graph adjacency list
def build_graph_alist(documents,cosine_matrix,t):
    graph = {}
    for i in range(len(documents)):
        id = str(i)
        graph[id] = []
        for j in range(len(cosine_matrix[i])):
            if cosine_matrix[i][j] >= t and i != j:
                graph[id].append(str(j))
    return graph

def text_to_sentences(text):
	text = text.replace('\n','').lower()
	text = text.replace('. ','.')
	text = text.replace('.','. ')
	tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

	text = '\n-----\n'.join(tokenizer.tokenize(text))
	return text.split('\n-----\n')

def stem_sentence(sentence):
    stemmer = PorterStemmer()
    load()
    stemmed = ""
    sentence = segment(sentence)
    for word in sentence:
        stemmed += stemmer.stem(word) + ' '
    return stemmed

#returns list of sentences
def stem_text(text):
    text = re.sub("([a-zA-Z0-9])’[a-zA-Z0-9]",r'\1',text)
    sentences = text_to_sentences(text)
    stemmed = []
    for sentence in sentences:
        stemmed.append(stem_sentence(sentence))
    return stemmed

#both args are text
def AP(systemSummaries, targetSummaries):
	systemSents = text_to_sentences(systemSummaries)
	targetSents = text_to_sentences(targetSummaries)

	AP = 0
	positives = 0
	total = 0
	for sent in systemSents:
		total += 1
		if sent in targetSents:
			positives += 1
			AP += positives/ total
	return AP/len(targetSents)

def remove_stopwords(text):
    return' '.join([word for word in text.split() if word not in cachedStopWords])