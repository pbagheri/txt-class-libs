# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 11:41:21 2019

@author: payam.bagheri
"""

# =============================================================================
# Libraries
# =============================================================================
import pandas as pd
from os import path
from tqdm import tqdm
import numpy as np
from difflib import SequenceMatcher
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords
from sklearn.utils import shuffle
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
# libraries needed for deep learning
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
import gensim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
import pickle
from sklearn.preprocessing import MultiLabelBinarizer

dir_path = path.dirname(path.dirname(path.abspath(__file__)))

# =============================================================================
# Loading google's word2vec
# =============================================================================
# Load Google's pre-trained Word2Vec model.
# model_w2v = gensim.models.KeyedVectors.load_word2vec_format('C:/Users/Payam/Downloads/GoogleNews-vectors-negative300.bin', binary=True)

# =============================================================================
# General Functions
# =============================================================================
# measures the similarity of two strings
def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

# cleans up the messaging responses
def mess_prep(mess):
    tokens = word_tokenize(str(mess))
    words = [word.lower() for word in tokens if word.isalpha()]
    words = [w for w in words if not w in stop_words]
    return words

def takeFirst(elem):
    return elem[0]

def isNaN(num):
    return num != num

# test example *******************
similar('the', 'they')

h = ['hormones',
'hermones',
'homorne',
'homorones',
'hormea',
'hormnr',
'hormonse',
'horome',
'horomons']

[similar(x,'hormones') for x in h]
# ********************************

def takeSecond(elem):
    return elem[1]


def softmax(l):
    summ = sum([np.exp(i) for i in l])
    sl = [np.exp(i)/summ for i in l]
    return sl

def get_vec(word):
    try:
        vec = model_w2v.get_vector(word)
    except KeyError:
        vec = np.zeros(300)
    return vec

# replaces a number with n is the number is above n
def if_replace(x,n):
    if x > n:
        r = n
    else:
        r = x
    return r

def str_join(lis):
    st = str()
    for i in lis:
        st = " ".join([st,i])
    return st

def weights_init_normal(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # get the number of the inputs
        n = m.in_features
        y = (1.0/np.sqrt(n))
        m.weight.data.normal_(0, y)
        m.bias.data.fill_(0)