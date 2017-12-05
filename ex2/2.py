import os
import numpy as np
import nltk

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


exec(open('../functions.py').read())
exec(open('priors.py').read())
exec(open('weights.py').read())


def build_graph_matrix(sentences,prior_func,weight_func):
    if not (callable(prior_func) and callable(weight_func)):
        return 'Not functions!'
    nsents = len(sentences)
    weights = np.zeros([nsents,nsents])
    priors = np.zeros(nsents)
    #create prior_weights
    for i in range(len(sentences)):
        priors[i] = prior_func(sentences[i],sentences)
    #create weights
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            weights[i][j] = weight_func(sentences[i],sentences[j],sentences)
    return (weights,priors)

def prestige(sentence_index,ranks,weights):
    sum = 0
    for j in range(len(weights[sentence_index])):
        sum += (ranks[j] * weights[sentence_index][j]) / np.sum(weights[j])
    return sum

def rank(weights,priors,itermax,damping):
    i = 0
    nsents = len(priors)
    pr = np.random.rand(nsents)
    
    while i < itermax:
        aux = pr
        for j in range(nsents):
            random = damping * (priors[j]/np.sum(priors))
            not_random = (1 - damping) * prestige(j,pr,weights)
            aux[j] = random + not_random
        pr = aux
        i += 1
    return pr

def get_top_n(array,n):
    indexes = np.argsort(array)
    top = {}
    for i in range(n):
        top[str(i)] = (array[indexes[i]])
    return top

def build_summary(sentences,prior_func,weight_func):

    box = build_graph_matrix(sentences,prior_func,weight_func)
    weights = box[0]
    priors = box[1]
    ranks = rank(weights,priors,50,0.15)
    top = get_top_n(ranks,5)
    summary = ""
    for i in top:
        summary += sentences[int(i[0])] + '\n'
    return summary


source_texts = os.listdir('../TeMario/source/')

MAP = 0
for text_file in source_texts:
	with open(r'../TeMario/source/' + text_file,'r',encoding='latin-1') as file:
	    text = file.read()
	sentences = text_to_sentences(text)
	summary = build_summary(sentences,uniform_prior,uniform_weight)
	with open(r'../TeMario/sums/' + 'Ext-' + text_file,'r',encoding='latin-1') as summary_file:
		MAP += AP(summary,summary_file.read())
MAP /= len(source_texts)

print("BASIC SUMMARY\n")
print('MAP:',MAP)