import sys
import nltk
nltk.download(['punkt', 'wordnet'])
import sqlite3
import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from train_classifier import load_data, tokenize



def test_load_data():
    X, Y, column_names = load_data("data/DisasterResponse.db")
    assert len(X) > 0, "Data not loaded"
    assert len(column_names) > 0, "Category names not loaded"
    
def test_tokenizer():
    tokens = tokenize("Package punkt is already up-to-date!")
    lst = ['package', 'punkt', 'is', 'already', 'up', 'to', 'date']
    assert tokens == lst, "Bad tokens"
    assert len(tokens) != 36, "Wrong no. of tokens"
    

if __name__ == "__main__":
    test_load_data()
    test_tokenizer()
    print("Everything passed")
