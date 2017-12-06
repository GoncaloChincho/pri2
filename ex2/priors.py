def uniform_prior(sent_index,graph,sentences):
    return 1/len(sentences)

#receives graph matrix for convenience...
def degree_centrality(sent_index,graph,sentences):
	links = graph[sent_index]
	nonzero = np.nonzero(links)
	return len(nonzero)/len(links)
