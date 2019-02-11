from Stats import Statistics
import os 
import pickle 
import csv 
from Test_Model_v7 import Test_Model

class Prediction(object):
    def __init__(self, patient_folder, model_name, trained_models_location):
        self.patient_folder = patient_folder
        self.model_name = model_name
        self.trained_models_location = trained_models_location
    
    def predict(self):
        get_all_patients_accuarcies={}
        awc, aw, ac, wc, a, w, c = [],[],[],[],[],[],[]
        for i in range(5, 37):
            if len(str(i)) ==1: 
                name = 'GOTOV0'+str(i)
            else:
                name = 'GOTOV'+str(i)     

            if os.path.exists(self.trained_models_location+self.model_name +'_'+name+'_.hdf5') == True: 
                print('Predicting test for patient', name)
                with open(self.patient_folder+name+'.pkl','rb') as f:
                    X_train, y_train, X_val, y_val, X_test, y_test, encoder = pickle.load(f)

                part = self.model_name+'_'+name
                test_model = Test_Model(self.trained_models_location, part)
                acc, awc1, aw1, ac1, wc1, a1, w1, c1 = test_model.evaluate_data(X_test, y_test, encoder, 8000)
                
                awc.append(awc1)
                aw.append(aw1)
                ac.append(ac1)
                wc.append(wc1)
                a.append(a1)
                w.append(w1)
                c.append(c1)

                if name in get_all_patients_accuarcies: 
                    get_all_patients_accuarcies[name].append(acc)
                else:
                    get_all_patients_accuarcies[name] = acc
            else:
                print('filepath is doesnt exist...try again :)')
        with open(self.model_name+'_prediction.csv', 'w') as f: 
            writer = csv.writer(f)
            for k, v in get_all_patients_accuarcies.items():
                writer.writerow([k, v])
                
        return awc, aw, ac, wc, a, w, c
    
    
    
    
    
    