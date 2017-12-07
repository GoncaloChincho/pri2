import numpy as np 

def uniform_prior(sent_index,graph,sentences):
    return 1/len(sentences)

#receives graph matrix for convenience...
def degree_centrality_prior(sent_index,graph,sentences):
	total_links = 0
	degree = 0 
	for i in range(len(graph)):
		nonzeros = len(np.nonzero(graph[i])[0])
		if i == sent_index:
			degree = nonzeros
		total_links += nonzeros
	if total_links == 0:
		return 0
	return degree/total_links


def sentence_position_prior(sent_index,graph,sentences):
	total = 0
	for i in range(len(sentences)):
		total += (i + 1)
	return (len(sentences) -  (sent_index)) / total