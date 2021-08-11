import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download(['punkt', 'wordnet', 'stopwords'])

import re
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


def tokenize(text):

    '''
    The function tokenizes input text.

    Input:
    text: text to be tokenized

    Output:
    tokens: tokens of input text, transfomations include:
        1) replacing urls with placeholder
        2) normalization
        3) removing punctuations
        4) tokenize
        5) removing stopwords
        6) lemmatization
    '''

    # replace urls with urlplaceholder
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")    
    
    # normalize and remove punctuations
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # tokenize
    tokens = word_tokenize(text)

    # remove stopwords and lemmatize
    tokens = [WordNetLemmatizer().lemmatize(word) for word in tokens if word not in stopwords.words("english")]
    
    return tokens


# implement TextLengthExtractor class to extract text length
class TextLengthExtractor(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):

        '''
        fit function returns self by default
        '''
        return self
    
    def transform(self, X):

        '''
        transform function returns length of each row of input as pandas DataFrame
        '''

        return pd.DataFrame(pd.Series(X).apply(len))
