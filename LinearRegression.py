#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 17:35:57 2020

@author: shusma
"""
import numpy as np

class LinearRegression():
    def __init__(self, X, y, alpha=0.03, n_iter=1500):
        self.alpha = alpha
        self.n_iter = n_iter
        self.n_samples = len(y)
        self.n_features = np.size(X,1)
        self.X = np.hstack((np.ones((self.n_samples,1)),(X - np.mean(X, 0))/np.std(X,0)))
        self.y = y[:, np.newaxis]
        self.params = np.zeros((self.n_features + 1, 1))
        self.coef_ = None
        self.intercept_ = None
        
    """ Fit linear model."""
    
    def fit(self):
        
        for i in range(self.n_iter):
            self.params = self.params - (self.alpha/self.n_samples) * \
            self.X.T @ (self.X @ self.params - self.y)
            
        self.intercept_ = self.params[0]
        self.coef_ = self.params[1:]
        
        return self 
    
    """ Return the coefficient of determination R^2 of the prediction. """
    
    def score(self, X=None, y=None):
        
        if X is None:
            X = self.X
        else:
            n_samples = np.size(X,0) #no. of rows 
            X = np.hstack((np.ones((n_samples,1)), (X - np.mean(X,0)) / np.std(X,0)))
        
        if y is None:
            y = self.y
        else:
            y = y[:, np.newaxis]
        
        y_pred = X @ self.params
        score = 1 - (((y - y_pred)**2).sum() / ((y - y.mean())**2).sum())
        
        return score
    
    """Predict using the linear model."""
    
    def predict(self, X):
        n_samples = np.size(X,0)
        y = np.hstack((np.ones((n_samples, 1)), (X - np.mean(X,0))/np.std(X,0))) @ self.params
        
        return y
    
    """Get parameters for this estimator."""
    
    def get_params(self):
         return self.params
        
        
        