import numpy as np 

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

exec(open('../functions.py').read())

#----------------------weights--------------------------------#
def is_edge(sent1_index,sent2_index,cosine_matrix,t):
	return (cos_sim(sent1_index,sent2_index,cosine_matrix) >= t) and (sent1_index != sent2_index)

def uniform_weight(sent1_index,sent2_index,sentences,cosine_matrix,t):
	if is_edge(sent1_index,sent2_index,cosine_matrix,t):
		return 1
	else:
		return 0

def cos_sim_weight(sent1_index,sent2_index,sentences,cosine_matrix,t):
	if is_edge(sent1_index,sent2_index,cosine_matrix,t):
		return cos_sim(sent1_index,sent2_index,cosine_matrix)
	else:
		return 0

