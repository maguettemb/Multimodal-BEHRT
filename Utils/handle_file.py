import pickle  
import pandas as pd
import os 

class handle_file():
    
    def __init__(self):
        return
    
    def _dump_pkl(self, file, filename):
        with open(filename+'.pkl', 'wb') as handle:
            pickle.dump(file, handle)
            print("The file has been saved in pkl format in {}".format(os.getcwd()))
            return 
      
    def _dump_csv(self, file, filename):
        pd.DataFrame.to_csv(file, filename+'.csv')
        print("The file has been saved in csv format in {}".format(os.getcwd()))
        return 
    
    def _load_pkl(self, filename):
        with open(filename, 'rb') as handle:
             return pickle.load(handle)
            
    def _load_pkl(self, filename):
        with open(filename, 'rb') as handle:
             return pd.read_pickle(handle)
        
    def _load_csv(self, filename):
        try: 
            return pd.read_csv(filename)
        except:
            print('filename must be in csv format')
        #return df

  
        
