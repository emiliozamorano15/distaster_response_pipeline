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

url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'


def load_data(database_filepath):
    '''
    Read data from local database
    INPUT:
        database_filepath: str path to local database file
    OUTPUT:
        table pandas dataframe
    '''
    conn = sqlite3.connect(database_filepath)
    
    df = pd.read_sql("SELECT * FROM tbl_disaster_response", conn)
    
    X = df.message.values
    df_Y = df.drop(['index', 'id', 'message', 'original', 'genre'], axis = 1)
    category_names = df_Y.columns
    Y = df_Y.values
    
    return X, Y, category_names
 

def tokenize(text):
    '''
    Obtain tokens from text. Transformations: to lower case, remove special characters,
    remove urls, and remove trailing and leading spaces. The clean text is then tokenized 
    by word and lemmatized.
    INPUTS:
        text: str string to be tokenized
    
    OUTPUTS: lst list of lemmatized tokens
    '''
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    
    text = re.sub(r"[^a-zA-Z0-9]", " ", text) 
    
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens
    


def build_model():
    ''' 
    Pipeline and grid definition for model building
    INPUTS:
    OUTPUTS: a GridSearchCV object from scikitlearn
        
    '''
    pipeline = Pipeline([
        ('text_features',
            Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])
        ),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
        'text_features__vect__ngram_range': ((1, 1), (1, 2), (2, 2)),
        'text_features__vect__max_df': (0.5, 0.75, 1.0),
        'text_features__vect_max_features': (None, 5000, 10000),
        'text_features__tfidf_use_idf': (True, False),
        'clf__n_estimators': [50, 100, 200],
        'clf__min_samples_split': [2, 5, 10]
    }
    
    cv = GridSearchCV(pipeline, param_grid=parameters, scoring = 'roc_auc')
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    pass


def save_model(model, model_filepath):
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
