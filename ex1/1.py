import nltk
import nltk.data
import sys

from nltk.stem.porter import *

from functions import build_graph_alist, text_to_sentences, AP,stem_text,remove_stopwords

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

def build_summary(text):
    vec = TfidfVectorizer()

    X = vec.fit_transform(sentences)
    S = cosine_similarity(X)

    graph = build_graph_alist(sentences,S,0.05)

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
stemmed_sentences = stem_text(text)
stem_stopwords = stem_text(remove_stopwords(text))

#Basic
basic_summary = build_summary(sentences)
stemmed_summary = build_summary(stemmed_sentences)
stopwords_summary = build_summary(stem_stopwords)

with open(target_filename,'r') as target_file:
    target_text = target_file.read()

print("BASIC SUMMARY\n")
print(basic_summary)
print("Average Precision Basic: ", AP(basic_summary,target_text))
print("\n######## STEMMED SUMMARY #######\n")
print(stemmed_summary)
print("Average Precision Stemmed: ", AP(stemmed_summary,target_text))
print("\n######## STEMMED STOPWORDS SUMMARY #######\n")
print(stopwords_summary)
print("Average Precision Stemmed: ", AP(stopwords_summary,target_text))










