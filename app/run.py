import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly import graph_objects
# from plotly.graph_objects import Bar
import joblib
from sqlalchemy import create_engine
from custom_transformer import tokenize, TextLengthExtractor


app = Flask(__name__)

# load data
engine = create_engine('sqlite:///data/DisasterResponse.db')
df = pd.read_sql_table('cleaned_data', engine)

# load model
model = joblib.load("models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # create visuals
    graphs = []

    # graph 1
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    graph1_data = []
    graph1_data.append(
        graph_objects.Pie(
            labels = genre_names,
            values = genre_counts,
            marker = {
                'colors': ['steelblue', 'darkseagreen', 'khaki'],
                'line':{
                    'color':'#FFFFFF', 
                    'width': 1
                }
            }
        )
    )

    graph1_layout = {
        'title': 'Distribution of Message Genres',
    }

    # graph 2
    english_counts = df[df['message']==df['original']].shape[0]
    other_counts = df.shape[0] - english_counts

    graph2_data = []
    graph2_data.append(
        graph_objects.Pie(
            labels = ['English', 'Other'],
            values = [english_counts, other_counts],
            marker = {
                'colors': ['indigo', 'lightgrey']
            }
        )
    )

    graph2_layout = {
        'title': 'Distribution of Message Languages'
    }

    # graph 3
    category_count = df.iloc[:,4:].sum(axis=0).sort_values(ascending=False)
    category_names = category_count.index
    category_names = [x.replace('_', ' ') for x in category_names]

    graph3_data = []
    graph3_data.append(
        graph_objects.Bar(
            x = category_names,
            y = category_count,
            marker = {
                'color': 'indianred'
            }
        )
    )

    graph3_layout = {
        'title': 'Distribution of Message Categories',
        'yaxis': {
            'title': "Count", 
            'type': "log"
        },
        'xaxis': {
            'title': "Category", 
            'tickangle': -45
        },
        'margin': {
            'b': 150
        }
    }


    graphs.append({'data': graph1_data, 'layout': graph1_layout})
    graphs.append({'data': graph2_data, 'layout': graph2_layout})
    graphs.append({'data': graph3_data, 'layout': graph3_layout})


    
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
    classification_results = dict(zip(df.columns[4:], classification_labels))

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