import sys
from os import listdir
from os.path import isfile, join
import re
import nltk
import nltk.data
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB
import functions
import numpy as np
debug = False

if '-d' in sys.argv:
    goal_source = sys.argv[(sys.argv).index('-d') + 1]
    goal_sums = sys.argv[(sys.argv).index('-d') + 2]
    train_source = sys.argv[(sys.argv).index('-d') + 3]
    train_sums = sys.argv[(sys.argv).index('-d') + 4]
else:
    goal_source = '.\\source'
    goal_sums = '.\\sums'
    train_source = '.\\train\\source'
    train_sums = '.\\train\\sums'

def cosine_similarity_matrix(doc, sentences):
    stop = nltk.corpus.stopwords.words('portuguese')
    vec = TfidfVectorizer(stop_words=stop)
    Y = vec.fit_transform([doc])
    X = vec.fit_transform(sentences)
    return cosine_similarity(X, Y)

def files_to_features(path):
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    features = []
    number_of_docs = 0
    for file in onlyfiles:
        fp = open(path + '\\' + file, 'r', encoding='latin-1')
        document = fp.read()
        fp.close()

        doc_lower = re.sub('(\w)(\\n)+', r'\1. ', document).lower()

        tokenizer = nltk.data.load('tokenizers/punkt/portuguese.pickle')
        doc = '\n-----\n'.join(tokenizer.tokenize(doc_lower))
        sentences = doc.split('\n-----\n')

        matrix = cosine_similarity_matrix(doc_lower, sentences)

        for i in range(0, len(sentences)):
            graph_cent = functions.degree_centrality(i, sentences, t=0.2)
            features.append([i, matrix[i][0], graph_cent])

        number_of_docs += 1
        if(debug):
            print("\n++++++++++++++++++++++\n")
            print("\n++++++++++++++++++++++\n")
            if(number_of_docs == 1):
                break

    return features

def files_to_class(source_path, sums_path):
    stop = set(stopwords.words('portuguese'))
    sourcefiles = [f for f in listdir(source_path) if isfile(join(source_path, f))]
    sumsfiles = [f for f in listdir(sums_path) if isfile(join(sums_path, f))]

    classification = []
    number_of_docs = 0
    i = 0
    for source, summary in zip(sourcefiles, sumsfiles):
        if(summary != "Sum-" + source):
            continue
        tokenizer = nltk.data.load('tokenizers/punkt/portuguese.pickle')
        fsrc = open(source_path + '\\' + source, 'r', encoding='latin-1')
        src_doc = re.sub('(\w)(\\n)+', r'\1. ', fsrc.read()).lower()
        src_doc = '\n-----\n'.join(tokenizer.tokenize(src_doc))
        src_sentences = src_doc.split('\n-----\n')
        fsrc.close()
        tokenizer = nltk.data.load('tokenizers/punkt/portuguese.pickle')
        fsum = open(sums_path + '\\' + summary, 'r', encoding='latin-1')
        sum_doc = re.sub('(\w)(\\n)+', r'\1. ', fsum.read()).lower()
        sum_doc = '\n-----\n'.join(tokenizer.tokenize(sum_doc))
        sum_sentences = sum_doc.split('\n-----\n')
        fsum.close()

        cl = [0 for k in range(len(src_sentences))]
        for i in range(0, len(src_sentences)):
            words_in_sum = 0
            for word in src_sentences[i].split():
                count = 0
                for smry in sum_sentences:
                    if(smry == src_sentences[i]):
                        cl[i] = 1
                        break
            classification.append(cl[i])

        number_of_docs += 1
        if(debug):
            print("\n++++++++++++++++++++++\n")
            print("\n++++++++++++++++++++++\n")
            if(number_of_docs == 1):
                break
    return classification

def files_to_sentences(path):
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    sentences = []
    for file in onlyfiles:
        fp = open(path + '\\' + file, 'r', encoding='latin-1')
        document = fp.read()
        fp.close()
        document = re.sub('(\w)(\\n)+', r'\1. ', document)
        tokenizer = nltk.data.load('tokenizers/punkt/portuguese.pickle')
        sentences.append(tokenizer.tokenize(document))
    return sentences

def train_perceptron(part_ft, cl):
    #naive-bayes features
    modelNB = train_naive_bayes(part_ft, cl)
    train_ft = get_features_with_NB(modelNB, part_ft)
    #target class
    if(len(train_ft) != len(cl)):
        print("Mismatch summaries and source files\n")
        return
    #train model
    model = Perceptron()
    model.fit(train_ft, cl)
    return model

def train_naive_bayes(train_ft, cl):
    if(len(train_ft) != len(cl)):
        print("Mismatch summaries and source files\n")
        return
    model = GaussianNB()
    model.fit(train_ft, cl)
    return model

def get_features_with_NB(model, features):
    probs = model.predict_proba(features)
    for i in range(0, len(features)):
        features[i].append(probs[i][0])
    return features


def summary():
    train_ft3 = files_to_features(train_source)
    cl = files_to_class(train_source, train_sums)
    #naive-bayes
    part_ft = files_to_features(goal_source)
    modelNB = train_naive_bayes(train_ft3, cl)
    features = get_features_with_NB(modelNB, part_ft)
    #perceptron
    model = train_perceptron(train_ft3, cl)
    labels = model.predict(features)

    #summary
    corpus = files_to_sentences(goal_source)
    candidates = []
    shift = 0
    d = 0
    for doc in corpus:
        s = []
        out = ""
        for i in range(0, len(doc)):
            if(labels[i+shift] == 1):
                s.append(doc[i])
                out += doc[i] + "\n"
                if(len(s) == 5):
                    break
        f = open('.\\outs\\out' + str(d) + '.txt', 'w', encoding='latin-1')
        f.write(out)
        f.close()
        candidates.append(s)
        shift = len(doc)
        d += 1
    return candidates

summaries = summary()
sum_text = ""
for doc in summaries:
    for sentence in doc:
        sum_text += sentence + "\n"

golden = files_to_sentences(goal_sums)
golden_text = ""
for doc in golden:
    for sentence in doc:
        golden_text += sentence + "\n"

MAP = functions.AP(golden_text, sum_text)
print(MAP)
