import pandas as pd 
import ast
import csv 

class Statistics(object): 
    
    def __init__(self, prediction_file_location, model_name):
        self.prediction_file_location = prediction_file_location
        self.model_name = model_name 
    
    def stats(self):
        dictionary = {}
        cols = ['awc', 'aw', 'ac', 'wc', 'a', 'w', 'c']
        with open(self.prediction_file_location, 'r') as f: 
            reader = csv.reader(f)
            for row in reader: 
                if row[0] in dictionary: 
                    dictionary[row[0]].append(ast.literal_eval(row[1]))
                else: 
                    dictionary[row[0]] = ast.literal_eval(row[1])
                    
        df = pd.DataFrame(dictionary)
        df=df.T
        df.columns = cols 
        print(df.mean())
        df.to_csv(self.model_name+'_stats.csv', index=True)