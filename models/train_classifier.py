  
import sys
import nltk
nltk.download(['punkt', 'wordnet'])
import sqlite3
import re
import numpy as np
import pandas as pd
import pickle
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer

url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'


def load_data(database_filepath):
    '''
    Read data from local database
    INPUT:
        database_filepath: str path to local database file
    OUTPUT:
        table pandas dataframe
    '''
    ## Read data from DB table
    conn = sqlite3.connect(database_filepath)
    df = pd.read_sql("SELECT * FROM tbl_disaster_response", conn)
    
    ## Split features and target 
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
    ## Remove urls if present
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    
    ## Remove special characters
    text = re.sub(r"[^a-zA-Z0-9]", " ", text) 
    
    ## tokenize text by words
    tokens = word_tokenize(text)
    
    ## Lemmatize each token
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens
    

def build_model():
    ''' 
    Pipeline and grid definition for model building
    INPUTS: none
    OUTPUTS: a GridSearchCV object from scikitlearn
        
    '''
    ## Create pipeline with two transformers and one estimator
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(
            RandomForestClassifier(n_jobs=-1)
            ))
    ])
    ## Define gridsearch hyperparameters
    parameters = {
        # 'vect__ngram_range': ((1, 1), (1, 2))
         'vect__max_df': (0.5, 1.0)
        # , 'vect__max_features': (None, 10000)
        , 'tfidf__use_idf': (True, False)
        , 'clf__estimator__n_estimators': [25, 50]
        , 'clf__estimator__min_samples_split': [2, 5, 10]
    }
    
    cv = GridSearchCV(pipeline, param_grid=parameters, cv= 3, verbose = 3)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluate model's performance on test data
    INPUTS:
        model: estimator
        X_test: feature variables
        Y_test: target variables
        category_names: target labels
    OUTPUT:
        metrics_df: pandas dataframe with performance metrics
    '''
    Y_pred = model.predict(X_test)
    
    scores = []
    
    # Calculate evaluation metrics for each category_name
    for i in range(len(category_names)):
        accuracy = accuracy_score(Y_test[:, i], Y_pred[:, i])
        precision = precision_score(Y_test[:, i], Y_pred[:, i], average = "micro")
        recall = recall_score(Y_test[:, i], Y_pred[:, i], average = "micro")
        f1 = f1_score(Y_test[:, i], Y_pred[:, i], average = "micro")
        
        scores.append([accuracy, precision, recall, f1])
  
    metrics = np.array(scores)
    metrics_df = pd.DataFrame(data = scores
                              , index = category_names
                              , columns = ['Accuracy', 'Precision', 'Recall', 'F1'])
                              
    print(metrics_df)
    return metrics_df
      


def save_model(model, model_filepath):
    '''
    save model as pickle object
    INPUTS:
        model: trained estimator
        model_filepath: str destinationm file        
    '''
    
    pickle.dump(model, open(model_filepath, 'wb'))


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

