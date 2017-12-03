import nltk
import nltk.data
import sys

from functions import build_graph_alist, text_to_sentences, AP

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from collections import Counter


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

def get_top_n(dictionary,n):
    d = Counter(dictionary)
    return d.most_common(n)

if len(sys.argv) > 1:
	filename = sys.argv[1]
	target_filename = sys.argv[2]
else:
	filename = 'text.txt'
	target_filename = 'textsum.txt'

file = open(filename,'r')


sentences = text_to_sentences(file.read())


vec = TfidfVectorizer()

X = vec.fit_transform(sentences)
S = cosine_similarity(X)

graph = build_graph_alist(sentences,S,0.05)

ranks = rank(graph,50,0.15)

top = get_top_n(ranks,5)
summary = ""
for i in top:
    summary += sentences[int(i[0])] + '\n'

print(summary)

target_file = open(target_filename,'r')

print("Average Precision: ", AP(summary,target_file.read()))






