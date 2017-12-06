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

def files_to_features(path):
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    features = []
    number_of_docs = 0
    for file in onlyfiles:
        if(file == "a.txt"):
            continue
        fp = open(path + '\\' + file, 'r')
        document = fp.read()
        doc_lower = re.sub('(\.)?(\n)+','. ',document).lower()
        vec = TfidfVectorizer()
        Y = vec.fit_transform([doc_lower])
        tokenizer = nltk.data.load('tokenizers/punkt/portuguese.pickle')
        doc_lower = '\n-----\n'.join(tokenizer.tokenize(doc_lower))
        sentences = doc_lower.split('\n-----\n')
        fp.close()

        X = vec.fit_transform(sentences)
        matrix = cosine_similarity(X, Y)

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
        f = open('output.txt', 'w+')

        cl = [0 for k in range(len(src_sentences))]
        for i in range(0, len(src_sentences)):
            words_in_sum = 0
            for word in src_sentences[i].split():
                count = 0
                for smry in sum_sentences:
                    if((smry in src_sentences[i]) or (src_sentences[i] in smry)):
                        if(smry == "."):
                            continue
                        cl[i] = 1
                        break
                    if((word in smry) and (word not in stop)):
                        count += 1
                if(count > len(sum_sentences)*0.2):
                    words_in_sum += 1
            if(words_in_sum > len(src_sentences[i])*0.5):
                cl[i] = 1
            classification.append(cl[i])
        f.close()
        number_of_docs += 1
        if(debug):
            print("\n++++++++++++++++++++++\n")
            print(source)
            print("\n++++++++++++++++++++++\n")
            if(number_of_docs == 1):
                break
    return classification

def train():
    features = files_to_features('.\\train\\source')
    cl = files_to_class('.\\train\\source', '.\\train\\sums')
    if(len(features) != len(cl)):
        print("Mismatch summaries and source files\n")
        return -1
    p = Perceptron()
    model = p.fit(features, cl)
    print(model.coef_)
    return 0

train()
# files_to_class('.\\train\\source', '.\\train\\sums')
