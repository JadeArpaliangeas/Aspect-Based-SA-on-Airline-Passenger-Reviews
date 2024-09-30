#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import spacy
from multiprocessing import Pool
import multiprocessing

nlp = spacy.load('en_core_web_sm')

def spacy_lemmatize(text):
    doc = nlp(text)
    return ' '.join([token.lemma_ for token in doc])

def lemmatize_sentences(sentences, num_processes=None):
    if num_processes is None:
        num_processes = max(1, multiprocessing.cpu_count() - 1)  # Use all cores except one
    with Pool(num_processes) as pool:
        lemmatized_sentences = pool.map(spacy_lemmatize, sentences)
    return lemmatized_sentences




