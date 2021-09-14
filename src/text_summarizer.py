import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import en_core_web_sm
import re
import heapq

import streamlit as st
import pandas as pd

import nltk
nltk.download('stopwords')
nltk.download('punkt')

def spacy_sum(text, no_sent):

    nlp = spacy.load('en_core_web_sm') # load the model (English) into spaCy

    doc = nlp(text) #pass the string doc into the nlp function.
    l = len(list(doc.sents)) #number of sentences in the given string
    st.write('Number of sentences in the original text: {}'.format(l) )

    #def generate_summary(text, no_sent):

    doc = nlp(text)
    sentence_list=[]
    for idx, sentence in enumerate(doc.sents): # we are using spacy for sentence tokenization
        sentence_list.append(re.sub(r'[^\w\s]','',str(sentence)))

    stopwords = nltk.corpus.stopwords.words('english')
        
    ## Word frequency 
    word_frequencies = {}  
    for word in nltk.word_tokenize(text):  
        if word not in stopwords:
            if word not in word_frequencies.keys():
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1

    maximum_frequncy = max(word_frequencies.values())

    for word in word_frequencies.keys():  
        word_frequencies[word] = (word_frequencies[word]/maximum_frequncy)

    ## Sentence scoring based on word frequency
    sentence_scores = {}  
    for sent in sentence_list:  
        for word in nltk.word_tokenize(sent.lower()):
            if word in word_frequencies.keys():
                if len(sent.split(' ')) < 30:
                    if sent not in sentence_scores.keys():
                        sentence_scores[sent] = word_frequencies[word]
                    else:
                        sentence_scores[sent] += word_frequencies[word]

    summary_sentences = heapq.nlargest(int(no_sent), sentence_scores, key=sentence_scores.get)
    summary = ''.join(summary_sentences)
    
    return summary
