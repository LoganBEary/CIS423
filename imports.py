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
    compare_list = list(set(self.column_list) - set(X.columns.to_list()))
    assert len(compare_list) == 0, f"{compare_list} not in table"
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
    return result
  
  
class PearsonTransformer(BaseEstimator, TransformerMixin):
  
  def __init__(self, threshold):
      self.threshold = threshold

  #define methods below
  def fit(self, X, y=None):
      print("Warning: PearsonTransformer.fit does nothing.")
      return X

  def transform(self, X, y=None):
      assert isinstance(X, pd.core.frame.DataFrame), f'PearsonTransformer.transform expected Dataframe but got {type(X)} instead.'
      assert isinstance(self.threshold, float), f'PearsonTransformer.transform expected a float but got {type(self.threshold)} instead'
      X_ = X.copy()
      X_ = X_.corr(method='pearson')
      X_ = X_.abs() > self.threshold
      upper_mask = np.triu(X_, 1)
      np.fill_diagonal(upper_mask, False)
      correlated_columns = [pos[1] for pos,x in np.ndenumerate(np.array(upper_mask)) if x == True]
      array_indx = [i for n, i in enumerate(correlated_columns) if i not in correlated_columns[:n]]
      correlated_columns = [masked_df.columns[x] for x in array_indx]
      X_ = X_.drop(columns=correlated_columns)
      return X_

  def fit_transform(self, X, y=None):
      result = self.transform(X)
      return result
    
class Sigma3Transformer(BaseEstimator, TransformerMixin):
  def __init__(self, target_column):  
    self.target_column = target_column

  def compute_3sigma_boundaries(df, column_name):
    #compute mean of column - look for method
    m = df[column_name].mean()
    #compute std of column - look for method
    sigma = df[column_name].std()
    return  (m-3*sigma,m+3*sigma) #(lower bound, upper bound)

  def fit(self, X, y=None):
    print("Warning: Sigma3Transformer.fit does nothing.")
    return X

  def transform(self, X):
    assert isinstance(X, pd.core.frame.DataFrame), f'expected Dataframe but got {type(X)} instead.'
    assert self.target_column in X.columns.to_list(), f'unknown column {self.target_column}'
    assert all([isinstance(v, (int, float)) for v in X[self.target_column].to_list()])
    X_ = X.copy()
    minb, maxb = compute_3sigma_boundaries(X_, self.target_column)
    X_[self.target_column] = X_[self.target_column].clip(lower=minb, upper=maxb)
    return X_

  def fit_transform(self, X, y=None):
    result = self.transform(X)
    return result
  
class TukeyTransformer(BaseEstimator, TransformerMixin):

  def __init__(self, target_column, fence='outer'):
    assert fence in ['inner', 'outer']
    self.target_column = target_column
    self.fence = fence

  def fit(self, X, y=None):
    print("Warning: TukeyTransformer.fit does nothing.")
    return X

  def transform(self, X, y=None):
    assert isinstance(X, pd.core.frame.DataFrame), f'expected Dataframe but got {type(X)} instead.'
    assert self.target_column in X.columns.to_list(), f'unknown column {self.target_column}'

    X_ = X.copy()
    q1 = X_[self.target_column].quantile(0.25)
    q3 = X_[self.target_column].quantile(0.75)
    iqr = q3-q1
    if self.fence == 'outer':
      outer_low = q1-3*iqr
      outer_high = q3+3*iqr
      X_[self.target_column] = X_[self.target_column].clip(lower=outer_low, upper=outer_high)

    else:
      inner_low = q1-1.5*iqr
      inner_high = q3+1.5*iqr
      X_[self.target_column] = X_[self.target_column].clip(lower=inner_low, upper=inner_high)
    
    return X_
  
  def fit_transform(self, X, y=None):
    result = self.transform(X)
    return result
  
  
