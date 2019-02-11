import os
import pickle 
from Models_v5 import Model
from Images import Image

class Train_model(object): 
    def __init__(self, epochs, n_batch, file_path, model_type): 
        self.epochs = epochs
        self.n_batch = n_batch
        self.file_path = file_path
        self.model_type = model_type
    
    @classmethod
    def train_model(cls,part, file_path, name, n_batch, epochs, model_type): 
        with open(file_path+name+'.pkl','rb') as f:
            X_train, y_train, X_val, y_val, X_test, y_test, encoder= pickle.load(f)
            print(X_train.shape, y_train.shape, X_val.shape, y_val.shape)

            
        model = Model(part, X_train, y_train, X_val, y_val, n_batch, epochs)
        if model_type == 'D': 
            history = model.Dense_basemodel()
        elif model_type == 'G': 
            history = model.GRU_basemodel()
        elif model_type == 'L': 
            history = model.LSTM_basemodel()
        elif model_type == 'T1': 
            history = model.try_1()
        elif model_type == 'T2': 
            history = model.try_2()
        elif model_type == 'T3': 
            history = model.try_3()
        elif model_type == 'T4': 
            history = model.try_4()
        elif model_type == 'T5': 
            history = model.try_5()
        elif model_type == 'T6': 
            history = model.try_6()
        elif model_type == 'T7': 
            history = model.try_7()
        elif model_type == 'MD': 
            history = model.model_deep()
     
        im = Image(history, part)
        im.create_image()
    
    def run_all(self):
        for i in range(5, 37):
            if i in [12, 16,19,23]:
                continue
            else:
                if len(str(i)) ==1: 
                    name = 'GOTOV0'+str(i)
                else:
                    name = 'GOTOV'+str(i)

            if os.path.exists(self.file_path+name+'.pkl') == True: 
                print('Starting to train model for patient', name)
                part =  self.model_type+'_'+ name
                self.train_model(part, self.file_path, name, self.n_batch, self.epochs, self.model_type)
            else: 
                print('filepath doesnt exist')

