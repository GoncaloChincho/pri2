import nltk
import nltk.data
import sys
import numpy as np

from nltk.stem.porter import *


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from collections import Counter

exec(open('../functions.py').read())

def build_graph_alist(documents,cosine_matrix,t):
    graph = {}
    for i in range(len(documents)):
        id = str(i)
        graph[id] = []
        for j in range(len(cosine_matrix[i])):
            if cos_sim(i,j,cosine_matrix) >= t and i != j:
                graph[id].append(str(j))
    return graph

def prestige(uid,ranks,links):
    sum = 0
    for vid in links[uid]:
        sum += ranks[vid] / len(links[vid])
    return sum

def rank(links,itermax,damping):
    pr = {}
    i = 0
    ndocs = len(links)
    
    for doc in links:
        pr[doc] = 1
        
    while i < itermax:
        aux = pr
        for doc in links:
            aux[doc] = (damping/ndocs) + ((1 - damping) * prestige(doc,pr,links))
        pr = aux
        i += 1
    return pr

def build_summary(sentences,t):
    S = get_cosine_similarities_matrix(sentences)

    graph = build_graph_alist(sentences,S,t)

    ranks = rank(graph,50,0.15)

    top = get_top_n(ranks,5)
    summary = ""
    for i in top:
        summary += sentences[int(i[0])] + '\n'
    return summary


def get_top_n(dictionary,n):
    d = Counter(dictionary)
    return d.most_common(n)

if len(sys.argv) > 1:
	filename = sys.argv[1]
	target_filename = sys.argv[2]
else:
	filename = 'text.txt'
	target_filename = 'textsum.txt'



with open(filename,'r') as file:
    text = file.read()

sentences = text_to_sentences(text)
#stemmed_sentences = stem_text(text)
#stem_stopwords = stem_text(remove_stopwords(text))

#Basic

#stemmed_summary = build_summary(stemmed_sentences)
#stopwords_summary = build_summary(stem_stopwords)

with open(target_filename,'r') as target_file:
    target_text = target_file.read()

results = []
summaries = []
tvals = np.arange(0.0, 1.05, 0.05)
for thresh in tvals:
    summary = build_summary(sentences,thresh)
    ap = AP(summary,target_text)
    results.append(ap)
    summaries.append(summary)

max = np.argmax(results)
print("Best summary with AP =",results[max],' for threshold =',tvals[max])
print("#-----------------------------------#")
print(summaries[max])


#print("\n######## STEMMED SUMMARY #######\n")
#print(stemmed_summary)
#print("Average Precision Stemmed: ", AP(stemmed_summary,target_text))
#print("\n######## STEMMED STOPWORDS SUMMARY #######\n")
#print(stopwords_summary)
#print("Average Precision Stemmed: ", AP(stopwords_summary,target_text))










