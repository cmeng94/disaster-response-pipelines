import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# build transformer to extract text length
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
