from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

exec(open('../functions.py').read())



def uniform_weight(sentence1,sentence2,sentences):
    return 1