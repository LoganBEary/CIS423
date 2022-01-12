import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


class DropColumnsTransformer(BaseEstimator, TransformerMixin):
  def __init__(self, column_list, action='drop'):
    assert action in ['keep', 'drop'], f'DropColumnsTransformer action {action} not in ["keep", "drop"]'
    assert isinstance(column_list, list), f'DropColumnsTransformer expected list but saw {type(column_list)}'
    self.column_list = column_list
    self.action = action

  #fill in rest below
  def fit(self, X, y=None):
    print("Warning: DropColumnsTransformer.fit does nothing.")
    return X

  def transform(self, X):
    X_ = X.copy()
    if self.action == 'keep':
      X_ = X_[self.column_list]
    else:
      X_ = X_.drop(columns=self.column_list)
    return X_

  def fit_transform(self, X, y=None):
    result = self.transform(X)
    return result
  
  
  class OHETransformer(BaseEstimator, TransformerMixin):
    def __init__(self, target_column, dummy_na=False, drop_first=True):  
      self.target_column = target_column
      self.dummy_na = dummy_na
      self.drop_first = drop_first
  
  #fill in the rest below

    def fit(self, X, y=None):
      print("Warning: OHETransformer.fit does nothing.")
    
      return X
  

    def transform(self, X_table):
      X_ = X_table.copy()
      X_ = pd.get_dummies(X_, 
                        prefix=self.target_column,
                        prefix_sep='_',
                        columns=[self.target_column],
                        dummy_na=self.dummy_na,
                        drop_first=self.drop_first)
      return X_

    def fit_transform(self, X, y=None):
      result = self.transform(X)
      return result
  
  
  #This class will rename one or more columns.
class RenamingTransformer(BaseEstimator, TransformerMixin):
  #your __init__ method below
  def __init__(self, renaming_dict:dict):  
    self.renaming_dict = renaming_dict
  #write the transform method without asserts. Again, maybe copy and paste from MappingTransformer and fix up.

  def fit(self, X, y=None):
    print("Warning: RenamingTransformer.fit does nothing.")
    return X

  def transform(self, X):
     # assert isinstance(X, pd.core.frame.DataFrame), f'MappingTransformer.transform expected Dataframe but got {type(X)} instead.'
     # assert self.mapping_column in X.columns.to_list(), f'MappingTransformer.transform unknown column {self.mapping_column}'
     X_ = X.copy()
     X_ = X_.rename(columns=self.renaming_dict)
     return X_

  def fit_transform(self, X, y=None):
    result = self.transform(X)
    return result
  
  
  class MappingTransformer(BaseEstimator, TransformerMixin):
  
  def __init__(self, mapping_column, mapping_dict:dict):  
    self.mapping_dict = mapping_dict
    self.mapping_column = mapping_column  #column to focus on

  def fit(self, X, y = None):
    print("Warning: MappingTransformer.fit does nothing.")
    return X

  def transform(self, X):
    assert isinstance(X, pd.core.frame.DataFrame), f'MappingTransformer.transform expected Dataframe but got {type(X)} instead.'
    assert self.mapping_column in X.columns.to_list(), f'MappingTransformer.transform unknown column {self.mapping_column}'
    X_ = X.copy()
    X_[self.mapping_column].replace(self.mapping_dict, inplace=True)
    return X_

  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result+
