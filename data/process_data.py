import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):

    '''
    The load_data function loads respective data sets.

    Input:
    messages_filepath: path to data set containing disaster response messages
    categories_filepath: path to data set containing categories of the messages

    Output:
    df: pandas DataFrame that merges the two input data sets on "id"
    '''

    # load data sets
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # merge data sets on "id"
    df = messages.merge(categories, on='id')
    return df

def clean_data(df):

    '''
    The function cleans the input data.

    Input:
    df: pandas DataFrame to be cleaned
    
    Output:
    df: the cleaned pandas DataFrame, steps include 
        1) splitting the categories column into separate columns
        2) convert values to binary
        3) remove duplicates
    '''

    # split categories columns
    categories = df['categories'].str.split(';', expand=True)
    row = categories.iloc[0]
    category_colnames = [x[:-2] for x in row]
    categories.columns = category_colnames

    # convert category values to binary
    for column in categories:
        categories[column] = categories[column].astype(str).str[-1]
        categories[column] = categories[column].astype(int)
        categories[column][categories[column] > 1] = 1
        categories[column][categories[column] < 0] = 0

    # concatenate to original data set
    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)

    # drop duplicates
    df.drop_duplicates(inplace=True)

    return df

def save_data(df, database_filename):

    '''
    The function saves cleaned data.
    Input:
    df: cleaned data
    database_filename: path for saved data set
    
    Output - None
    '''

    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('cleaned_data', engine, if_exists='replace', index=False)  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()