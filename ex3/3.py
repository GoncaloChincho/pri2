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
debug = False

def cosine_similarity_matrix(doc, sentences):
    vec = TfidfVectorizer()
    Y = vec.fit_transform([doc])
    X = vec.fit_transform(sentences)
    return cosine_similarity(X, Y)

def files_to_features(path):
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    features = []
    number_of_docs = 0
    for file in onlyfiles:
        fp = open(path + '\\' + file, 'r')
        document = fp.read()
        fp.close()

        doc_lower = re.sub('(\.)?(\n)+', '. ', document).lower()

        tokenizer = nltk.data.load('tokenizers/punkt/portuguese.pickle')
        doc = '\n-----\n'.join(tokenizer.tokenize(doc_lower))
        sentences = doc.split('\n-----\n')

        matrix = cosine_similarity_matrix(doc_lower, sentences)

        for i in range(0, len(matrix)):
            features.append([i, matrix[i][0]])

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
        fsrc = open(source_path + '\\' + source, 'r')
        src_doc = re.sub('(\.)?(\n)+','. ', fsrc.read()).lower()
        src_doc = '\n-----\n'.join(tokenizer.tokenize(src_doc))
        src_sentences = src_doc.split('\n-----\n')
        fsrc.close()
        tokenizer = nltk.data.load('tokenizers/punkt/portuguese.pickle')
        fsum = open(sums_path + '\\' + summary, 'r')
        sum_doc = re.sub('(\.)?(\n)+','. ', fsum.read()).lower()
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
        fp = open(path + '\\' + file, 'r')
        document = fp.read()
        fp.close()
        document = re.sub('(\.)?(\n)+', '. ', document)
        tokenizer = nltk.data.load('tokenizers/punkt/portuguese.pickle')
        sentences.append(tokenizer.tokenize(document))
    return sentences

def train():
    train_ft = files_to_features('.\\train\\source')
    cl = files_to_class('.\\train\\source', '.\\train\\sums')
    if(len(train_ft) != len(cl)):
        print("Mismatch summaries and source files\n")
        return
    model = Perceptron()
    model.fit(train_ft, cl)
    return model

def summary():
    model = train()
    features = files_to_features('.\\source')
    labels = model.predict(features)
    corpus = files_to_sentences('.\\source')
    candidates = []
    shift = 0
    for doc in corpus:
        s = []
        for i in range(0, len(doc)):
            if(labels[i+shift] == 1):
                s.append(doc[i])
                if(len(s) == 5):
                    break
        candidates.append(s)
        shift = len(doc)
    return candidates

f = open('output.txt', 'a')
summaries = summary()
for s in summaries:
    for sentence in s:
        f.write(sentence)
        f.write("\n")
f.close()
