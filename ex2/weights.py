import numpy as np 

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

exec(open('../functions.py').read())

def is_edge(sent1_index,sent2_index,cosine_matrix,t):
	return cos_sim(sent1_index,sent2_index,cosine_matrix) > t

def uniform_weight(sent1_index,sent2_index,sentences,cosine_matrix,t):
	if is_edge(sent1_index,sent2_index,cosine_matrix,t):
		return 1
	else:
		return 0

#receives graph matrix for convenience...
def degree_centrality(sent_index,graph):
	links = graph[sent_index]
	nonzero = np.nonzero(links)
	return len(nonzero)
