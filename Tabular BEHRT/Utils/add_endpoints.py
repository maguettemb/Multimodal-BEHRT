import pandas as pd
import pickle 


def open_end(rfs = 5):
    
    if rfs == 5:
        filename = '/home/jovyan/work/Workdir/Late intergation/Structured data/data/Data extraction /endpoints_5y.pkl'
    elif rfs == 10:
        filename = '/home/jovyan/work/Workdir/Late intergation/Structured data/data/Data extraction /endpoints_10y.pkl'
    with open(filename, 'rb') as handle:
             return pickle.load(handle)
        

class add_endp():
    def __init__(self, df, rfs = 5):
        self.df = df
        self.rfs = rfs
    
    def merge(self):
        if 'Num_dossier' in list(self.df.columns):
            df = self.df.rename(columns={"Num_dossier" : "numdos_curie"})
            df_merge = df.merge(open_end(rfs = self.rfs), on='numdos_curie', how='left')
         
        else:
            df_merge = self.df.merge(open_end(rfs = self.rfs), on='numdos_curie', how = 'left')   
            
        return df_merge.rename(columns={"numdos_curie":"Num_dossier"})
            
        
