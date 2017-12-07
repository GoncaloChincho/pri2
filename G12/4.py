import nltk
import nltk.data
import sys
import nltk
import re
import copy
import string

from nltk.stem.porter import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from bs4 import BeautifulSoup  
from nltk.stem import PorterStemmer

sys.path.append('..')
from functions import *

def text_to_sentences(text):
    text = re.sub('(\w)(\\n)+', r'\1. ',text)
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    text = '\n-----\n'.join(tokenizer.tokenize(text))
    return text.split('\n-----\n')

def build_graph_alist(documents,cosine_matrix,t):
    graph = {}
    for i in range(len(documents)):
        id = str(i)
        graph[id] = []
        for j in range(len(cosine_matrix[i])):
            if cosine_matrix[i][j] >= t and i != j:
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

def get_top_n(dictionary,n):
    d = Counter(dictionary)
    return d.most_common(n)

def remove_img_tags(data):
    p = re.compile(r'<img.*?/>')
    return p.sub('', data)

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, porter)
    return stems

sources = ['cnn', 'latimes', 'nytimes', 'washington']
news = ""
doc_links = []
sentences = [] 

#----------------------------- Parsing XML -----------------------------#
for source in sources:
    with open('worldnews/' + source + '.xml', encoding = "utf8") as f:
        soup = BeautifulSoup(f, 'xml')
        items = soup.find_all('item')

    for item in items:
        news = item.find('title').get_text() + '\n'
        news += remove_img_tags(item.find('description').get_text()).replace('<p>', '').replace('</p>', '').replace('</a>','')  + '\n'
        sentences.append(news)
   
        if source == sources[0]:
            doc_links.append(item.find('feedburner:origLink').get_text())
        else:
            doc_links.append(item.find('link').get_text())
#----------------------------------------------------------------------#


#---------- Generating an HTML page illustrating the summary ----------#
def build_summary(sentences):
    vec = TfidfVectorizer(tokenizer=tokenize, stop_words='english')
    
    X = vec.fit_transform(sentences)
    S = cosine_similarity(X)
 
    graph = build_graph_alist(sentences,S,.01)

    ranks = rank(graph,50,0.5)

    top = get_top_n(ranks,5)
    
    summary = """<h1>World News</h1><br>"""
    for i in top:
        summary += """<a href=" """+doc_links[int(i[0])]+""" " target="_blank">"""+document[int(i[0])] + """</a><br><br>"""
    return summary

document = copy.deepcopy(sentences)
porter = PorterStemmer()
token_dict = {}
remove = dict.fromkeys(map(ord, string.punctuation))
xpto=[]

for sentence in document:
    lowers = sentence.lower()
    no_punctuation = lowers.translate(remove).replace('\n',' ')
    xpto.append(no_punctuation)

basic_summary = build_summary(xpto)

with open('page.html', 'w') as f:
    f.write(basic_summary)
#----------------------------------------------------------------------#