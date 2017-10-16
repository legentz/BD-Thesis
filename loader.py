#-*- coding: utf-8 -*-
from sklearn.externals import joblib

class Loader:
    def __init__(self, paths=None):
        self.paths = paths

    def get_data(self):
    	print '--> Loading datasets'
    	
        dicts = joblib.load(self.paths['dicts'])
        train = joblib.load(self.paths['train'])
        dev = joblib.load(self.paths['dev'])
        test = joblib.load(self.paths['test'])

        return dicts, train, dev, test
