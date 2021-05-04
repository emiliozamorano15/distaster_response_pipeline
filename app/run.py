import json
import plotly
import numpy as np
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Pie
from plotly.express import colors

# from sklearn.externals import joblib
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('tbl_disaster_response', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    categories = df.drop(['original', 'index', 'id', 'message', 'genre'], axis = 1)
    categories_count = categories.sum().sort_values().values
    categories_names = categories.sum().sort_values().index
    
    ## pie chart with counts per genre

    trace1 = Pie(
        labels = genre_names
        , values = genre_counts
        , marker_colors = colors.qualitative.Alphabet_r
    )
    
    ## horizontal barchart with counts per category

    trace2 = Bar(
        x = categories_count
        , y = categories_names
        , orientation = 'h')
    
    ## append both graphs
    graphs = [
        {'data': [trace1],
        'layout':{
            'title':'Distribution of Message Genres'
        }
        },
        {'data': [trace2],
        'layout':{
            'title': 'Distribution of Categories',
            'yaxix':{'title':'Count'},
            'xaxis':{'title':'Category'}
        }
        }
            ]
        
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[5:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()