import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from custom_transformer import tokenize, TextLengthExtractor
import pickle

def load_data(database_filepath):

    '''
    The function loads cleaned data set.

    Input:
    database_filepath: path to cleaned data set

    Output:
    X: column of loaded data set containing messages
    Y: columns of loaded data set containing categories of messages
    category_names: names of categories
    '''

    # load cleaned data
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('cleaned_data', engine)

    # separate into messages and categories
    X = df['message'].values
    Y = df.iloc[:,4:]

    # get category names
    category_names = Y.columns

    return X, Y, category_names

def build_model():

    '''
    The function builds the machine learning model.

    Input: 
    None

    Output:
    pipeline: ML pipeline
    '''

    pipeline = Pipeline([

    	# build feature union of text transformation and text length extractor
        ('features', FeatureUnion([
            
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())                        
            ])),   
            
            ('text_length', TextLengthExtractor())

        ])),

        # random forest classifier
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

#    I understand how to tune parameters using GridSearchCV (please refer to ML_pipeline.ipynb and classification results therein in main folder), but for runtime considerations, I prefer not to run it in terminal.

#    parameters = {
#        'clf__estimator__n_estimators': [25, 50],
#        'clf__estimator__max_depth': [300, 400],
#        'clf__estimator__min_samples_split': [4, 6],
#    }
#
#    cv = GridSearchCV(pipeline, para_grid = parameters)
#    return cv

    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):

    '''
    The function evaluates the model.

    Input:
    model: model to be evaluated
    X_text: test messages to be classified
    Y_text: categories of test messages
    category_names: names of classification categories

    Output:
    None
    '''

	# get predicted categories
    Y_pred = model.predict(X_test)

    # print test results
    print(classification_report(Y_test, Y_pred, target_names = category_names))


def save_model(model, model_filepath):

    '''
    The function saves constructed machine learning model as pickle file.
    
    Input:
    model: model to be saved
    model_filepath: path for saved model
    
    Output:
    None
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
