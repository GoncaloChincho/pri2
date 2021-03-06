import os
import numpy as np
import nltk
import sys

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


exec(open('../functions.py').read())
exec(open('priors.py').read())
exec(open('weights.py').read())

def build_graph_matrix(sentences,weight_func,t):
     if not callable(weight_func):
         return 'Not functions!'
     nsents = len(sentences)
     weights = np.zeros([nsents,nsents])
     
     cos_matrix = get_cosine_similarities_matrix(sentences)
     #create weights
     for i in range(len(sentences)):
         for j in range(len(sentences)):
             weights[i][j] = weight_func(i,j,sentences,cos_matrix,t)
     return weights

def build_priors(sentences,graph,prior_func):
	if not callable(prior_func):
		return 'Not functions!'
	nsents = graph.shape[0]
	priors = np.zeros(nsents)

	for i in range(nsents):
		priors[i] = prior_func(i,graph,sentences)
	return priors



def prestige(sentence_index,ranks,weights):
    sum = 0
    for j in range(len(weights[sentence_index])):
        if weights[sentence_index,j]== 0:
            continue
        else:   
            sum += ((ranks[j] * weights[j][sentence_index]) / np.sum(weights[j]))
    return sum

def rank(weights,priors,itermax,damping):
    i = 0
    nsents = len(priors)
    pr = np.ones(nsents)
    
    while i < itermax:
        aux = pr
        for j in range(nsents):
            random = damping * priors[j]
            if random != 0:
                random /= np.sum(priors)
            not_random = (1 - damping) * prestige(j,pr,weights)
            aux[j] = random + not_random
        pr = aux
        i += 1
    
    return pr

def get_top_n(array,n):
    indexes = sorted(range(len(array)), key=lambda i: array[i], reverse=True)[:n]
    top = {}
    for i in range(n):
        top[str(indexes[i])] = (array[indexes[i]])
    return top

def build_summary(sentences,prior_func,weight_func,t):
    weights = build_graph_matrix(sentences,weight_func,t)
    priors = build_priors(sentences,weights,prior_func)
    ranks = rank(weights,priors,50,0.15)
    top = get_top_n(ranks,5)
    summary = ""
    for i in top:
        summary += sentences[int(i)] + '\n'
    return summary

if '-d' in sys.argv:
	source_path = sys.argv[(sys.argv).index('-d') + 1]
	sums_path = sys.argv[(sys.argv).index('-d') + 2]
else: 
	source_path = '../TeMario/source/'
	sums_path = '../TeMario/sums/'

if '-p' in sys.argv:
    priorf = globals()[sys.argv[(sys.argv).index('-p') + 1]]
else:
    priorf = sentence_position_prior

if '-w' in sys.argv:
    weightf = globals()[sys.argv[(sys.argv).index('-w') + 1]]
else:
    weightf = cos_sim_weight

results = []
summaries = []
tvals = np.arange(0.0, 1.05, 0.05)

source_texts = os.listdir(source_path)

for thresh in tvals:
    MAP = 0

    for text_file in source_texts:
        with open(source_path + text_file,'r',encoding='Latin-1') as file: #source_path + text_file
            text = file.read()
        sentences = text_to_sentences(text)
        summary = build_summary(sentences,priorf,weightf,thresh)
        with open(sums_path+ 'Ext-' + text_file ,'r',encoding='Latin-1') as summary_file: #sums_path+ 'Ext-' + text_file    '../ex1/textsum.txt'
            MAP += AP(summary,summary_file.read())
    MAP /= len(source_texts)
    results.append(MAP)
max = np.argmax(results)
print("Best MAP =",results[max],' for threshold =',tvals[max])