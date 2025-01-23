from datetime import date, datetime
import numpy as np
import os
from datetime import date, datetime, timedelta
import pandas as pd
from Early_integration.Utils.handle_file import handle_file
hf = handle_file()
import unicodedata

def clean_text(text):
    
    ## remove accents
    text = str(text)
    normalized = unicodedata.normalize("NFKD", text)
    normalized = "".join([c for c in normalized if not unicodedata.combining(c)])
        
    ## remove white space
    
    normalized = normalized.strip()
    normalized = normalized.lower()
        
        
    return normalized 

def date_to_str(d):
    
    stringdate = d.strftime("%Y%m%d")
    
    return stringdate


            
def str_to_date(d):
            
    try:
        if isinstance(d, str):
            try: 
                date = datetime.strptime(d, '%Y%m%d')
            except:
                pass

            try:
                date = datetime.strptime(d, '%d/%m/%Y')
            except:
                pass

            try: 
                date = datetime.strptime(d, '%d-%m-%Y')
            except:
                pass

            try: 
                date = datetime.strptime(d, '%Y-%m-%d')
            except:
                pass

        elif pd.isna(d):
            date = pd.NaT

        else:
            date = d

    except TypeError: 
        print('TypeError: Dates should be in date format or in str, not {} '.format(type(d)))
    
    return date
    
      
def cal_age(date_event, date_diagnosis, age, age_round = True): 
    
    """this function calculates the age gab between the day of the diagnose and the current date, 
    and then the age for the current date. 
    If age_round = True: The age DOESN'T take into account delays in days and months. Therefore new_age type is int
    If age_round = False : The age takes into account delays in days and months. Therefore new_age type is float"""
    date_event = str_to_date(date_event)
    date_diagnosis = str_to_date(date_diagnosis)
    
    if age_round == True:
        try:
        
            delta_years = date_event.year - date_diagnosis.year
            new_age = np.round(age + delta_years)
    
        except TypeError: 
            print('TypeError: age should be a float')
    else:
   
        try:
            ## most recent date - less recent date 
            delta_days = (date_event - date_diagnosis).days
            delta_years = (delta_days /30.4375) / 12
            new_age = round(age + delta_years, 3)


        except:
            new_age = age
    
    return int(new_age)


def delay(d1, d2, unit):
    
    """This function calculates the delay between 2 dates in months, in weeks or in days
    d1, d2: datetime
    """
    
    d1 = str_to_date(d1)
    d2 = str_to_date(d2)
    
    if unit == 'month':
        return int(np.ceil((d1 - d2).days / 30.4375))   ## use the round number for the delay vocabulary in MLM and the classification
    if unit == 'week':
        return int(np.ceil((d1 - d2).days / 7))
    if unit == 'day':
        return int((d1 - d2).days) 
    


def time_delay(df, unit = 'month'):
    
    a = [0]
    for i, row in df.iterrows():

        d1 = row['Date_exam']
        try: 
            d2 = df.loc[i+1,'Date_exam']
        except KeyError:
            break

        a.append(delay(d2, d1, unit = unit))
    
    return a

def int_to_list(integer):
    """Convert a integer to a list of integer"""
    return [integer]


def cal_therapies_0(data, number_of_cycles, timestamps = 'regular', therap_window = True, window = 14):

    """ This function makes the therapies sequential using the date they've been administrated to display their code
    surgery : 1, 'ht':2, 'rt':3, 'ct':4, 'antiher2':5 
    data : the data with the therapies in binary form
    number_of_cycles: number of cycles of cures for treatments  # cf. 'nb_cycles_adj_ct'
    timestamps: The regularity of therapy session, each 5days: regular vs irregular
    ts: the number of days between two treatments or the sequence of days when it's irregular eq:[2, 5, 7]: 2, 5 and 7 days after the first treatment
    # Take into account the date_end of the treatment in the sequence ts 
    window : The number of days to add before and after the treatment in order to be matched with reports writen before and after the treatment"""

    therapies = ['breast_surgery', 'ht', 'rt', 'ct', 'antiher2']
    dates = ['dat_first_surg', 'dat_first_ht', 'dat_first_rt', 'dat_first_ct', 'dat_first_antiher2']
    config = {
        'breast_surgery':1, 'ht':2, 'rt':3, 'ct':4, 'antiher2':5, 'no therapy': 0
    }
    
    ts = {'breast_surgery':0, 'ct': 6, 'rt': 6, 'ht': 6, 'antiher2':6}
    # The timestamps between cures: 
    
    data = data.sort_values(by=['Num_dossier','Date_exam'],  ignore_index= True)
    data['therapies'] =  [[0] for r in range(len(data))]
    
    for i, row in data.iterrows():                  # for each row => each date of medical reports

        list_of_dates = []
        list_of_therapies = []
        dict_therap = dict()
        
        for therapy, date in zip(therapies, dates):
            if row[therapy] != 0:                       ## gather all the therapies into a dict with the id and the current date (date_report)

                dict_therap['Num_dossier'] = row['Num_dossier']

                dict_therap[therapy] = row[therapy]

                date_therapy = datetime.strptime(row[date], '%Y-%m-%d')
                dict_therap[date] = date_therapy

                Date_exam = datetime.strptime(row['Date_exam'], '%Y%m%d')
                dict_therap['date_event'] = Date_exam

                list_of_therapies.append(therapy) 
                list_of_dates.append(date_therapy)  

        list_of_therapies.append('Date_event')          ## gather the list of therapies and the dates for each row 
        list_of_dates.append(Date_exam)   


        sorted_dates = sorted(zip(list_of_dates, list_of_therapies))         ## sort the dates from the less recent to the most recent event 
        
        if sorted_dates[0][1] == 'Date_event':          
            ## If the less recent event between therapies and the current day, we don't have therapy yet = 0
            ## I need to see if within the BC window we will find the date_event (in a backward way)
 
            for j in range(len(sorted_dates)):  
                if sorted_dates[j][1] != 'Date_event':
                    window_dates = [sorted_dates[j][0]+timedelta(days=i) for i in range(-window, window+1)]
                   
                    if dict_therap['date_event'] in window_dates:
                        data.loc[i, 'therapies'].append(config[sorted_dates[j][1]])
                     #   data.at[i, 'therapies'] += int_to_list(config[sorted_dates[j][1]])
                      
    

        for j in range(len(sorted_dates)):  
            if sorted_dates[j][1] != 'Date_event':
                if timestamps == 'regular':
                    seq_of_dates = [sorted_dates[j][0]+timedelta(days=n*ts[sorted_dates[j][1]]) for n in range(number_of_cycles)]
                elif timestamps == 'irregular':
                    seq_of_dates = [date1]+[date1+timedelta(days=n) for n in ts[sorted_dates[j][1]]]

                if therap_window == True:
                    seq_of_dates = [seq_of_dates[i]+timedelta(days=j) for j in range(-window, window+1) for i in range(len(seq_of_dates))]

                if sorted_dates[j][1] == 'breast_surgery':
                    seq_of_dates = [sorted_dates[j][0]+timedelta(days=i) for i in range(-window, window+1)]


                if dict_therap['date_event'] in seq_of_dates:
                    data.loc[i, 'therapies'].append(config[sorted_dates[j][1]])
                   # data.at[i, 'therapies'] += int_to_list(config[sorted_dates[j][1]])
    
    ## return list with no duplicates
    data.loc[:,'therapies'] = data['therapies'].apply(lambda x: list(dict.fromkeys(x)))
    
    ## remove 0 when there is therapies in the list 
    
    data.loc[:,'therapies'] = data['therapies'].apply(lambda x:  x[1:] if len(x)>1 else x)
 
    return data 


#@jit(nopython=True, parallel=True)
def cal_subtherapies(df, fill_with = 'string'):
    
    data = df.copy()
    ## I use therapies in dict to have access to the code directly when using data 
    if fill_with == 'int':
        therapies = {'breast_surgery':1, 'ht':2, 'rt':3, 'ct':4, 'adj_antiher2':5, 'no therapie': 0}
        subtherapies = {therapies['breast_surgery']: {(2, 1): 1, (2, 2): 2, (2, 3): 3, (2, 0): 4, (1, 1): 5, (1, 2): 6, (1, 3): 7, (1, 0): 8}, 
                  therapies['ht']: {1: 9, 2: 10, 3: 11}, therapies['rt']: {1: 12}, therapies['ct']: {1:13, 2:14, 3: 15, 'nan' :16} , therapies['adj_antiher2']:{1: 17}
        }
        data['sub_therapies'] =  [[0] for r in range(len(data))]
        
        
    elif fill_with == 'string':
        therapies = {
        'breast_surgery':'chirurgie', 'ht':'hormone therapie', 'rt': 'radio_therapie', 'ct':'chimiotherapie', 'adj_antiher2':'adjuvant antiher2', 'no therapy': 'pas de therapie'
    }
            
        subtherapies = {therapies['breast_surgery']: {(2, 1): 'mastectomie et biopsie du ganglion sentinelle', 
                                                        (2, 2): 'mastectomie et dissection ganglionnaire axillaire', 
                                                        (2, 3): 'mastectomie et BGS + DSA', 
                                                        (2, 0): 'mastectomie seule', 
                                                        (1, 1): 'tumorectomie et biopsie du ganglion sentinelle', 
                                                        (1, 2): 'tumorectomie et dissection ganglionnaire axillaire', 
                                                        (1, 3): 'tumorectomie et BGS + DSA', 
                                                        (1, 0): 'tumorectomie seule'}, 
                  therapies['ht']: {1: 'tamoxifene', 
                                    2: 'inhibiteur aromatase', 
                                    3: 'autres'},
                          therapies['rt']: {1: 'radio_therapie'}, 
                          therapies['ct']: {1:'anthra-taxanes', 
                                            2:'anthra',
                                            3: 'taxanes', 
                                            4: 'autres',
                                            'nan' : 'non renseigné'} , 
                          therapies['adj_antiher2']:{1: 'antiher2 adjuvant'}
        }
        data['sub_therapies'] =  [['pas de subtherapie'] for r in range(len(data))]
    
    # breast surgery : (x, y) : y:(1: Sentinel Node Biopsy, 2:  Axillary node dissection, 3: Both SND and AND, 0: no axillar surgery)
    # x: (1: Lumpectomy, 2: Mastectomy)
    # ht :( 1: tamoxifen, 2: aromatase, 3: others)
    # ct : 1: anthra-taxanes, 2: anthra, 3: others, nan: missing 
    
    
    ## columns needed : 'axillary_surgery_4cl', 'breast_surgery', ht_type_3cl, 'adj_ct_regimen'
    cols = {therapies['breast_surgery']: ['breast_surgery_3cl', 'axillary_surgery_4cl'], therapies['ht']: 'ht_type_3cl', 
            therapies['rt']:'rt', therapies['ct']: 'adj_ct_regimen', therapies['adj_antiher2']:'adj_antiher2'}
    
   
    
    for i, row in data.iterrows():                  # for each row => each date of medical reports
        try: 
            if row['therapies'] != ['no therapie']:
                for j in row['therapies']: 
                    if j == therapies['breast_surgery']:  # if it's breast surgery
                        data.loc[i, 'sub_therapies'].append(subtherapies[j][(row[cols[j][0]],row[cols[j][1]])])
                        #data.at[i, 'sub_therapies'] += int_to_list(subtherapies[j][(row[cols[j][0]],row[cols[j][1]])])

                    else: 

                        if pd.isna(row[cols[j]]) :
                            data.loc[i, 'sub_therapies'].append(subtherapies[j]['nan'])
                        else:
                            data.loc[i, 'sub_therapies'].append(subtherapies[j][row[cols[j]]])
                        #data.at[i, 'sub_therapies'] += int_to_list(subtherapies[j][row[cols[j]]])

        except:
            data.loc[i, 'sub_therapies'].append('pas de sous therapie')
            
    ## return list with no duplicates
    data.loc[:,'sub_therapies'] = data['sub_therapies'].apply(lambda x: list(dict.fromkeys(x)))
    
    ## remove 0 when there is therapies in the list 
    
    data.loc[:,'sub_therapies'] = data['sub_therapies'].apply(lambda x:  x[1:] if len(x)>1 else x)
 
    return data 


class spark_init(object):
    def __init__(self, sparkConfig=None, name='Multimod_BEHRT'):
        self._setup_spark(sparkConfig)
        self.sc, self.sqlContext = self._init_spark(name=name)

    def _setup_spark(self, sparkConfig):

        if sparkConfig == None:
            config = {'memory': '300g', 'excutors': '4', 'exe_mem': '50G', 'result_size': '80g',
                      'temp': '/home/yikuan/tmp', 'offHeap':'16g'}
        else:
            config = sparkConfig

        os.environ["PYSPARK_PYTHON"] = "" # python spark path
        pyspark_submit_args = ' --driver-memory ' + config['memory'] + ' --num-executors ' + config['excutors'] + \
                              ' --executor-memory ' + config['exe_mem']+ \
                              ' --conf spark.driver.maxResultSize={} --conf spark.memory.offHeap.size={} --conf spark.local.dir={}'\
                                  .format(config['result_size'], config['offHeap'], config['temp']) +\
                              ' pyspark-shell'

        os.environ["PYSPARK_SUBMIT_ARGS"] = pyspark_submit_args

    def _init_spark(self, name='Multimod_BEHRT'):
        sc = pyspark.SparkContext(appName=name)
        sqlContext = SQLContext(sc)
        sqlContext.sql("SET spark.sql.parquet.binaryAsString=true")
        return sc, sqlContext
    
def read_txt(sc, sqlContext, path):
    """read from txt to pyspark dataframe"""
    file = sc.textFile(path)
    head = file.first()
    content = file.filter(lambda line: line != head).map(lambda k: k.split('\t'))
    df = sqlContext.createDataFrame(content, schema=head.split('\t'))
    return df


def read_parquet(sqlContext, path):
    """read from parquet to pyspark dataframe"""
    return sqlContext.read.parquet(path)

def read_csv(sqlContext, path):
    """read from parquet to pyspark dataframe"""
    return sqlContext.read.csv(path, header=True)

def int_to_str(x):
    return [str(int(f)) for f in x]

def float_to_str(list_of_int):
    float_to_int = [ int(x) for x in list_of_int]
    return [str(x) for x in float_to_int]

def flatten_list(x):
    try:
        res = [item for sublist in x for item in sublist]
    except:
        res = x
    return res

def flatten_array(x):
    x = np.hstack(x)
    x = x.flatten()
    return x



def cal_therapies_fr(df, therap_window = True, window = 14):

    """ This function makes the therapies sequential using the date they've been administrated to display their code
    surgery : 1, 'ht':2, 'rt':3, 'ct':4, 'antiher2':5 
    data : the data with the therapies in binary form
    number_of_cycles: number of cycles of cures for treatments  # cf. 'nb_cycles_adj_ct'
    timestamps: The regularity of therapy session, each 5days: regular vs irregular
    ts: the number of days between two treatments or the sequence of days when it's irregular eq:[2, 5, 7]: 2, 5 and 7 days after the first treatment
    # Take into account the date_end of the treatment in the sequence ts 
    window : The number of days to add before and after the treatment in order to be matched with reports writen before and after the treatment"""
    df = data.copy()
    
    therapies = ['breast_surgery', 'ht', 'rt', 'ct', 'adj_antiher2']
    dates = ['dat_first_surg', 'dat_first_ht', 'dat_first_rt', 'dat_first_ct', 'dat_first_antiher2']
    config = {
        'breast_surgery':1, 'ht':2, 'rt':3, 'ct':4, 'adj_antiher2':5, 'no therapy': 0
    }
    
    ts = {'breast_surgery':0, 'ct': 6, 'rt': 6, 'ht': 6, 'adj_antiher2':6}
    # The timestamps between cures: 
    
    data = df.copy()
    data['therapies'] =  [[0] for r in range(len(data))]
    
    for i, row in data.iterrows():                  # for each row => each date of medical reports
        
     #   print("... row "+str(i)+" ...")
       # print(data.loc[i, 'therapies'])
        list_of_dates = []
        list_of_therapies = []
        dict_therap = dict()
        
        #data.at[i, 'therapies'] = data.apply(lambda x:[], axis=1)
        for therapy, date in zip(therapies, dates):
            if row[therapy] != 0:                       ## gather all the therapies into a dict with the id and the current date (date_report)
                try: 
                    dict_therap['Num_dossier'] = row['Num_dossier']

                    dict_therap[therapy] = row[therapy]

                    date_therapy = datetime.strptime(row[date], '%Y-%m-%d')
                    dict_therap[date] = date_therapy

                    Date_exam = datetime.strptime(row['Date_exam'], '%Y%m%d')
                    dict_therap['date_event'] = Date_exam

                    list_of_therapies.append(therapy) 
                    list_of_dates.append(date_therapy)  
               
                except:
                    pass

        list_of_therapies.append('Date_event')          ## gather the list of therapies and the dates for each row 
        list_of_dates.append(Date_exam)   


        sorted_dates = sorted(zip(list_of_dates, list_of_therapies))         ## sort the dates from the less recent to the most recent event 
  
        if sorted_dates[0][1] == 'Date_event':          
            ## If the less recent event between therapies and the current day, we don't have therapy yet = 0
            ## I need to see if within the BC window we will find the date_event (in a backward way)
 
            for j in range(len(sorted_dates)):  
                if sorted_dates[j][1] != 'Date_event':
                    window_dates = [sorted_dates[j][0]+timedelta(days=i) for i in range(-window, window+1)]
                   
                    if dict_therap['date_event'] in window_dates:
                       # data.loc[i, 'therapies'].append(config[sorted_dates[j][1]])
                        data.at[i, 'therapies'] += int_to_list(config[sorted_dates[j][1]])
                      
                  #  else:
                     #   data.at[i, 'therapies'] = int_to_list(0)
               
      
        for j in range(len(sorted_dates)):  
            if sorted_dates[j][1] != 'Date_event':
                
                if sorted_dates[j][1] == 'breast_surgery':
                    seq_of_dates = [sorted_dates[j][0]+timedelta(days=i) for i in range(-window, window+1)]

                
                if sorted_dates[j][1] == 'ht':
               
                    # For ht it will be 1 injection per day for 5, 7-10 years for patients with lymph nodes affected
                    if row['pnuicc_4cl'] == 1 or pd.isna(row['pnuicc_4cl']) == True:  # if 0 nodes involved or missing information (520patients)
                        
                        seq_of_dates =  pd.date_range(start = sorted_dates[j][0], end = sorted_dates[j][0]+timedelta(days=5*365.2425))
                    else:  
                        seq_of_dates =  pd.date_range(start = sorted_dates[j][0], end= sorted_dates[j][0]+timedelta(days=8.5*365.2425))
                
                if sorted_dates[j][1] == 'rt':
               
                    # 1 cure per day for 4-6 weeks (5 is choosen)
                    seq_of_dates =  pd.date_range(start = sorted_dates[j][0], end= sorted_dates[j][0]+timedelta(weeks=5))
                
                if sorted_dates[j][1] == 'adj_antiher2':
             
                    # 18 injections each 3 weeks (combined with CT) (verifie ca)
                    seq_of_dates = [sorted_dates[j][0]+timedelta(days=n*21) for n in range(18)]
                
                if sorted_dates[j][1] == 'ct':
                   
                    if row['adj_ct_regimen'] == 1 or row['adj_ct_regimen'] == 2  :
                        delay = sorted_dates[j][0]
                        seq_of_dates = []
                        try:
                            for i in range(row['nb_cycles_adj_ct']):
                                seq_of_dates = seq_of_dates+[delay+timedelta(days=n*21) for n in range(4)]
                                delay = seq_of_dates[-1]+timedelta(weeks=3+3+11)  # CF TIMELINE => 3weeks after we start the taxol cure, then 1cure per week for 11 weeks 
                                                                                    # and 3 more weeks between cycles (21d + 77d + 21d)
                        except:
                            seq_of_dates = [sorted_dates[j][0]+timedelta(days=n*21) for n in range(4)]
                        
                    if row['adj_ct_regimen'] == 1:
                        delay = sorted_dates[j][0]
                        seq_of_dates = []
                        try:
                            for i in range(row['nb_cycles_adj_ct']):
                                seq_of_dates = seq_of_dates+[delay+timedelta(days=n*7) for n in range(11)]
                                delay = seq_of_dates[-1]+timedelta(days=21)
                        except:
                            seq_of_dates = [sorted_dates[j][0]+timedelta(days=n*7) for n in range(11)]
                
                 ### Problem: I don't have the information regarding the molecule used: Let's assume that taxotere or taxol is used
                 ## 
                 #   if molecules == 'taxol':
                 #       seq_of_dates = [sorted_dates[j][0]+timedelta(days=n*7) for n in range(11)]
                 #   if molecules == 'taxotère':
                 #       seq_of_dates = [sorted_dates[j][0]+timedelta(days=n*21) for n in range(4)]      
                
                if therap_window == True:
                    seq_of_dates = [seq_of_dates[i]+timedelta(days=j) for j in range(-window, window+1) for i in range(len(seq_of_dates))]

                if dict_therap['date_event'] in seq_of_dates:
                    data.at[i, 'therapies'] += int_to_list(config[sorted_dates[j][1]])
               
                    
    ## return list with no duplicates
    data.loc[:,'therapies'] = data['therapies'].apply(lambda x: list(dict.fromkeys(x)))
    
    ## remove 0 when there is therapies in the list 
    
    data.loc[:,'therapies'] = data['therapies'].apply(lambda x:  x[1:] if len(x)>1 else x)
 
    return data 

def divide_frame(data, n_patient, n):
    """ this function allows us to split the huge dataframe of reports into small ones."""
    
    list_of_frames = []
    list_of_chunks = list(divide_chunks(n_patient, n))
    grouped = data.groupby(data.Num_dossier)
    
    for i in list_of_chunks:
        list_of_frames.append(pd.concat([grouped.get_group(name) for name in i]))
    
    return list_of_frames

def divide_chunks(l, n):
    """This function splits the list of patients into list of n patients that will be used as patients for each subgroup of dataframe"""
      
    # looping till length l
    for i in range(0, len(l), n): 
        yield l[i:i + n]

def fill_modalities(sequence):
    
    """ this function creates an new embedding that shows. the different modality.
    """
    
    mod_list = ['MOD']*len(sequence)
    modalities = ['DEB', 'biologique', 'clinique', 'rapports medicaux', 'FIN']
    ## <=>.       ['CLS', CA153', 'MONO','LEUC', 'PN', 'LYMP', 'therapies', 'sub_therapies', 'Service', 'Category', 'SEP']
    out = []
    k = 0
    
    
    for i, j in zip(sequence, mod_list):
              
        if i == 'SEP':
            k+=1
            out.append(modalities[k])
        elif i == 'CLS':
            out.append(modalities[k])
            k+=1  
        elif i != j:
               out.append(modalities[k])     
        else:
            k+=1
        
    return out 

def cal_therapies(df, therap_window = True, window = 14, fill_with = 'string'):

    """ This function makes the therapies sequential using the date they've been administrated to display their code
    surgery : 1, 'ht':2, 'rt':3, 'ct':4, 'antiher2':5 
    data : the data with the therapies in binary form
    number_of_cycles: number of cycles of cures for treatments  # cf. 'nb_cycles_adj_ct'
    timestamps: The regularity of therapy session, each 5days: regular vs irregular
    ts: the number of days between two treatments or the sequence of days when it's irregular eq:[2, 5, 7]: 2, 5 and 7 days after the first treatment
    # Take into account the date_end of the treatment in the sequence ts 
    window : The number of days to add before and after the treatment in order to be matched with reports writen before and after the treatment"""

    data = df.copy()
    therapies = ['breast_surgery', 'ht', 'rt', 'ct', 'adj_antiher2']
    dates = ['dat_first_surg', 'dat_first_ht', 'dat_first_rt', 'dat_first_ct', 'dat_first_antiher2']
    
    if fill_with == 'int':
        config = {
            'breast_surgery':1, 'ht':2, 'rt':3, 'ct':4, 'adj_antiher2':5, 'no therapy': 0
        }
        data['therapies'] =  [[0] for r in range(len(data))]
        
    elif fill_with == 'string':
        config = {'breast_surgery':'chirurgie', 'ht':'hormone therapie', 'rt':'radio_therapie', 'ct':'chimiotherapie', 'adj_antiher2':'adjuvant antiher2', 'no therapy': 'pas de therapie'}
        data['therapies'] =  [['pas de therapie'] for r in range(len(data))]
        
  
    for i, row in data.iterrows():                  # for each row => each date of medical reports
        
        list_of_dates = []
        list_of_therapies = []
        dict_therap = dict()
       # print(row['
        #data.at[i, 'therapies'] = data.apply(lambda x:[], axis=1)
        for therapy, date in zip(therapies, dates):
            if row[therapy] != 0:                       ## gather all the therapies into a dict with the id and the current date (date_report)
                try: 
                    dict_therap['Num_dossier'] = row['Num_dossier']

                    dict_therap[therapy] = row[therapy]
                    
                    date_therapy = str_to_date(row[date])
                  #  date_therapy = datetime.strptime(row[date], '%Y%m%d')
                    dict_therap[date] = date_therapy

                    Date_exam = str_to_date(row['Date_exam'])
                    #Date_exam = datetime.strptime(row['Date_exam'], '%Y%m%d')
                    dict_therap['date_event'] = Date_exam

                    list_of_therapies.append(therapy) 
                    list_of_dates.append(date_therapy)
               
                except:
                    pass

        list_of_therapies.append('Date_event')          ## gather the list of therapies and the dates for each row 
        list_of_dates.append(Date_exam)   

        try:
            sorted_dates = sorted(zip(list_of_dates, list_of_therapies))         ## sort the dates from the less recent to the most recent event 
        except: 
            print(row['Num_dossier'])
            
        if sorted_dates[0][1] == 'Date_event':          
            ## If the less recent event between therapies and the current day, we don't have therapy yet = 0
            ## I need to see if within the BC window we will find the date_event (in a backward way)
 
            for j in range(len(sorted_dates)):  
                if sorted_dates[j][1] != 'Date_event':
                    window_dates = [sorted_dates[j][0]+timedelta(days=i) for i in range(-window, window+1)]
                   
                    if dict_therap['date_event'] in window_dates:
                       # data.loc[i, 'therapies'].append(config[sorted_dates[j][1]])
                        data.at[i, 'therapies'] += int_to_list(config[sorted_dates[j][1]])
                      
                  #  else:
                     #   data.at[i, 'therapies'] = int_to_list(0)
               
      
        for j in range(len(sorted_dates)):  
            if sorted_dates[j][1] != 'Date_event':
                
                if sorted_dates[j][1] == 'breast_surgery':
                    seq_of_dates = [sorted_dates[j][0]+timedelta(days=i) for i in range(-window, window+1)]

                
                if sorted_dates[j][1] == 'ht':
               
                    # For ht it will be 1 injection per day for 5, 7-10 years for patients with lymph nodes affected
                    if row['pnuicc_4cl'] == 1 or pd.isna(row['pnuicc_4cl']) == True:  # if 0 nodes involved or missing information (520patients)
                        
                        seq_of_dates =  pd.date_range(start = sorted_dates[j][0], end = sorted_dates[j][0]+timedelta(days=5*365.2425))
                    else:  
                        seq_of_dates =  pd.date_range(start = sorted_dates[j][0], end= sorted_dates[j][0]+timedelta(days=8.5*365.2425))
                
                if sorted_dates[j][1] == 'rt':
               
                    # 1 cure per day for 4-6 weeks (5 is choosen)
                    seq_of_dates =  pd.date_range(start = sorted_dates[j][0], end= sorted_dates[j][0]+timedelta(weeks=5))
                
                if sorted_dates[j][1] == 'adj_antiher2':
             
                    # 18 injections each 3 weeks (combined with CT) (verifie ca)
                    seq_of_dates = [sorted_dates[j][0]+timedelta(days=n*21) for n in range(18)]
                
                if sorted_dates[j][1] == 'ct':
                   
                    if row['adj_ct_regimen'] == 1 or row['adj_ct_regimen'] == 2  :
                        delay = sorted_dates[j][0]
                        seq_of_dates = []
                        try:
                            for i in range(row['nb_cycles_adj_ct']):
                                seq_of_dates = seq_of_dates+[delay+timedelta(days=n*21) for n in range(4)]
                                delay = seq_of_dates[-1]+timedelta(weeks=3+3+11)  # CF TIMELINE => 3weeks after we start the taxol cure, then 1cure per week for 11 weeks 
                                                                                    # and 3 more weeks between cycles (21d + 77d + 21d)
                        except:
                            seq_of_dates = [sorted_dates[j][0]+timedelta(days=n*21) for n in range(4)]
                        
                    if row['adj_ct_regimen'] == 1:
                        delay = sorted_dates[j][0]
                        seq_of_dates = []
                        try:
                            for i in range(row['nb_cycles_adj_ct']):
                                seq_of_dates = seq_of_dates+[delay+timedelta(days=n*7) for n in range(11)]
                                delay = seq_of_dates[-1]+timedelta(days=21)
                        except:
                            seq_of_dates = [sorted_dates[j][0]+timedelta(days=n*7) for n in range(11)]
                
                 ### Problem: I don't have the information regarding the molecule used: Let's assume that taxotere or taxol is used
                 ## 
                 #   if molecules == 'taxol':
                 #       seq_of_dates = [sorted_dates[j][0]+timedelta(days=n*7) for n in range(11)]
                 #   if molecules == 'taxotère':
                 #       seq_of_dates = [sorted_dates[j][0]+timedelta(days=n*21) for n in range(4)]      
                
                if therap_window == True:
                    seq_of_dates = [seq_of_dates[i]+timedelta(days=j) for j in range(-window, window+1) for i in range(len(seq_of_dates))]

                if dict_therap['date_event'] in seq_of_dates:
                    data.at[i, 'therapies'] += int_to_list(config[sorted_dates[j][1]])
               
                    
    ## return list with no duplicates
    data.loc[:,'therapies'] = data['therapies'].apply(lambda x: list(dict.fromkeys(x)))
    
    ## remove 0 when there is therapies in the list 
    
    data.loc[:,'therapies'] = data['therapies'].apply(lambda x:  x[1:] if len(x)>1 else x)
 
    return data 


def clean_accents(df):
    for i, row in df.iterrows(): 

        for inputs in ['inputs_quantiles', 'inputs_normal_range', 'inputs_curve_intersections',
       'inputs_bis_quantiles', 'inputs_bis_normal_range', 'inputs_bis_curve_intersections'] :
            row[inputs] = [i.replace('Ã©', 'é') for i in row[inputs]] 
            row[inputs] = [i.replace('Ã¨', 'è') for i in row[inputs]] 
            row[inputs] = [i.replace('Ã´', 'o') for i in row[inputs]] 
            row[inputs] = [i.replace('ô', 'o') for i in row[inputs]] 
            
            
    for col in ['age', 'delays']:
        df[col] = df[col].apply(lambda x: float_to_str(x))
   
    return df


def get_index_positions(list_of_elems, element):
    ''' Returns the indexes of all occurrences of give element in
    the list- listOfElements '''
    index_pos_list = []
    index_pos = 0
    while True:
        try:
            # Search for item in list from indexPos to the end of list
            index_pos = list_of_elems.index(element, index_pos)
            # Add the index position in list
            index_pos_list.append(index_pos)
            index_pos += 1
        except ValueError as e:
            break
    return index_pos_list

def add_sep(list_to_change, element= 'CLS', change_with = 'SEP'):
    
    indexes = get_index_positions(list_to_change, element)
    
    for i, j in zip(indexes[1:], range(len(indexes)-1)):
        list_to_change.insert(i+j, change_with)
        
    return list_to_change



def categorize_delays(delay):
    cat_delay = dict()
    for week in range(4):
        cat_delay[week] = 'W'+str(week)
    for month in range(1,13):
        cat_delay[int(np.round(month*4.3482))] = 'M'+str(month)
    ## for delay more thana year 
    cat_delay[1000] = 'LT' # Long term

    if delay in cat_delay.keys():
        return cat_delay[delay]
    elif delay <= 52: 
        keys_array = np.array(list(cat_delay.keys()))
        return cat_delay[keys_array[keys_array>delay][0]]
    else: 
        return cat_delay[1000]
        

def set_grade(row):
    if pd.isna(row):
        return 'Missing'
    elif row == 'Non précisé':
        return 'Missing'
    elif row == 'Non évaluable': 
        return 'Missing'
    else:
        return row

def set_nodes(x, y):
    if pd.isna(x) & pd.isna(y):
        return 'Missing'
    elif pd.isna(x):
        return y
    elif pd.isna(y):
        return x
    elif pd.isna(x) == False & pd.isna(y) == False:
        return x+y
    
      
def set_modes(data, var_1, var_2):
    df = (data.groupby(var_1)[var_2]
               .apply(lambda x: x.mode().iat[0])
               .reset_index())
    return df

def set_size(data, x, how='clinical'):
    if how == 'clinical':
        df = set_modes(data, 'TUICC', 'TCLIN')
        if pd.isna(x.TCLIN) == False:
            return x.TCLIN
        elif pd.isna(x.TUICC) == False: 
            return df.loc[df['TUICC']==x.TUICC]['TCLIN'].values[0]
        else:
            return 'Missing'
    
    elif how == 'pathological':
        df = set_modes(data, 'PTUICC', 'TINF')
        if pd.isna(x.TINF) == False:
            return x.TINF
        elif pd.isna(x.PTUICC) == False:
            return df.loc[df['PTUICC']==x.PTUICC]['TINF'].values[0]
        else:
            return 'Missing'
        
def set_tumor_size(row):
    if row.clinical_ts != 'Missing':
        if row.Date_exam < row.dat_first_surg:
            return row.clinical_ts/10 # en cm 
        else:
            if row.pathological_ts != 'Missing':
                return row.pathological_ts/10 #en cm 
            else:
                return 'Missing'
    else:
        return row.clinical_ts
        
def set_subtype(row):
    if pd.isna(row) == False:
        return row
    else:
        return 'Missing'    
    
def categorize_size(size):
    
    bins = list(np.arange(0.0, 30.0, .5))
    labels = list()
    for i in bins[1:]:
        labels.append('S'+str(i))
    return pd.cut([size], bins=bins, labels=labels, include_lowest=True)
  
def categorize_nodes_num(nodes):
    bins = list(range(-1, 50))
    labels = list()
    for i in bins[1:]:
        labels.append('N'+str(i))
  
    return pd.cut([nodes], bins=bins, labels=labels, include_lowest=True)
    
def categorize_grade(grade):
    if grade != 'Missing':
        return 'G'+str(grade)
    else:
        return grade
    
def categorize_st(st):
    if st == 1:
        return 'Luminal'
    elif st == 2:
        return 'TNBC'
    elif st == 3:
        return 'HER2+/RH+'
    elif st == 4:
        return 'HER2+/RH-'
    
    

def fill_modalities_for_twos(sequence):
    
    """ this function creates an new embedding that shows. the different modality.
    """
    
    mod_list = ['MOD']*len(sequence)
   # modalities = ['biologique', 'clinique', 'rapports medicaux', 'FIN']
    modalities = ['CA153','MONO', 'LEUC', 'PN', 'LYMP', 'CA153','MONO', 'LEUC', 'PN', 'LYMP', 'age', 'grade', 'node', 'size', 'subtype', 'therapie','rapports medicaux', 'FIN']
    ## <=>.       ['CLS', CA153', 'MONO','LEUC', 'PN', 'LYMP', 'therapies', 'sub_therapies', 'Service', 'Category', 'SEP']
    out = []
    k = 0
    for i, j in zip(sequence, mod_list):
        
        if i == 'SEP':
            k+=1
            out.append(modalities[k])
           # k=0
        elif i == 'CLS':
            out.append('DEB')
          #  k+=1  
        elif i != j:
            out.append(modalities[k])     
        else:
            k+=1
        
    return out 

  
def fill_modalities(sequence):
    
    """ this function creates an new embedding that shows. the different modality.
    """
    
    mod_list = ['MOD']*len(sequence)
   # modalities = ['biologique', 'clinique', 'rapports medicaux', 'FIN']
    modalities = ['CA153','MONO', 'LEUC', 'PN', 'LYMP', 'age', 'grade', 'node', 'size', 'subtype', 'therapie', 'rapports medicaux', 'FIN']
    out = []
    k = 0
    for i, j in zip(sequence, mod_list):
        
        if i == 'SEP':
            k+=1
            out.append(modalities[k])
            k=0
        elif i == 'CLS':
            out.append('DEB')
          #  k+=1  
        elif i != j:
            out.append(modalities[k])     
        else:
            k+=1
        
    return out 



def delete_multiple_element(list_object, indices):
    indices = sorted(indices, reverse=True)
    for idx in indices:
        if idx < len(list_object):
            list_object.pop(idx)
            
def get_indexes(seq):
    list_of_idx = list()
    for idx in range(len(seq)):
        if seq[idx] == 'pas de therapie' or seq[idx] == 'pas de sous therapie' or seq[idx] == '0.0' or seq[idx] == 'Missing':
            list_of_idx.append(idx)
    return list_of_idx

def remove_unhappened_events(seq, seq2):
    idxs = get_indexes(seq)
    new_seq = [i for j, i in enumerate(seq2) if j not in idxs]
    return new_seq

## col + preprocessed

def remove_no_event(df, inputs_col, other_col, report='report'):
    
    for other_col in [other_col]:
        df.loc[:,other_col+'_preprocessed'] = df.apply(lambda row : remove_unhappened_events(row['inputs_normal_range'], row[other_col]), axis = 1)
        
    for col in inputs_col:
        df.loc[:,col+'_preprocessed'] = df.apply(lambda row : remove_unhappened_events(row[col], row[col]), axis = 1)

        return df


def clean_text(text):
    
    ## remove accents
    text = str(text)
    normalized = unicodedata.normalize("NFKD", text)
    normalized = "".join([c for c in normalized if not unicodedata.combining(c)])
        
    ## remove white space
    
    normalized = normalized.strip()
    normalized = normalized.lower()
        
        
    return normalized 

def date_to_str(d):
    
    stringdate = d.strftime("%Y%m%d")
    
    return stringdate

def delay(d1, d2, unit = 'month'):
    
    """This function calculates the delay between 2 dates in months, in weeks or in days
    d1, d2: datetime
    """
    

    if unit == 'month':
        return (d1 - d2).dt.days / 30.4375 
    if unit == 'week':
        return (d1 - d2).dt.days / 7
    if unit == 'day':
        return (d1 - d2).dt.days 
    

    
    
def cal_age(date_event, date_diagnosis, age, age_round = True): 
    
    """this function calculates the age gab between the day of the diagnose and the current date, 
    and then the age for the current date. 
    If age_round = True: The age DOESN'T take into account delays in days and months. Therefore new_age type is int
    If age_round = False : The age takes into account delays in days and months. Therefore new_age type is float"""
    date_event = str_to_date(date_event)
    date_diagnosis = str_to_date(date_diagnosis)
    
    if age_round == True:
        try:
        
            delta_years = date_event.year - date_diagnosis.year
            new_age = np.round(age + delta_years)
    
        except TypeError: 
            print('TypeError: age should be a float')
    else:
   
        try:
            ## most recent date - less recent date 
            delta_days = (date_event - date_diagnosis).days
            delta_years = (delta_days /30.4375) / 12
            new_age = round(age + delta_years, 3)


        except:
            new_age = age
    
    return int(new_age)


def delay(d1, d2, unit):
    
    """This function calculates the delay between 2 dates in months, in weeks or in days
    d1, d2: datetime
    """
    
    d1 = str_to_date(d1)
    d2 = str_to_date(d2)
    
    if unit == 'month':
        return int(np.ceil((d1 - d2).days / 30.4375))   ## use the round number for the delay vocabulary in MLM and the classification
    if unit == 'week':
        return int(np.ceil((d1 - d2).days / 7))
    if unit == 'day':
        return int((d1 - d2).days) 
    


def time_delay(df, unit = 'month'):
    
    a = [0]
    for i, row in df.iterrows():

        d1 = row['Date_exam']
        try: 
            d2 = df.loc[i+1,'Date_exam']
        except KeyError:
            break

        a.append(delay(d2, d1, unit = unit))
    
    return a

def int_to_list(integer):
    """Convert a integer to a list of integer"""
    return [integer]


def cal_therapies_0(data, number_of_cycles, timestamps = 'regular', therap_window = True, window = 14):

    """ This function makes the therapies sequential using the date they've been administrated to display their code
    surgery : 1, 'ht':2, 'rt':3, 'ct':4, 'antiher2':5 
    data : the data with the therapies in binary form
    number_of_cycles: number of cycles of cures for treatments  # cf. 'nb_cycles_adj_ct'
    timestamps: The regularity of therapy session, each 5days: regular vs irregular
    ts: the number of days between two treatments or the sequence of days when it's irregular eq:[2, 5, 7]: 2, 5 and 7 days after the first treatment
    # Take into account the date_end of the treatment in the sequence ts 
    window : The number of days to add before and after the treatment in order to be matched with reports writen before and after the treatment"""

    therapies = ['breast_surgery', 'ht', 'rt', 'ct', 'antiher2']
    dates = ['dat_first_surg', 'dat_first_ht', 'dat_first_rt', 'dat_first_ct', 'dat_first_antiher2']
    config = {
        'breast_surgery':1, 'ht':2, 'rt':3, 'ct':4, 'antiher2':5, 'no therapy': 0
    }
    
    ts = {'breast_surgery':0, 'ct': 6, 'rt': 6, 'ht': 6, 'antiher2':6}
    # The timestamps between cures: 
    
    data = data.sort_values(by=['Num_dossier','Date_exam'],  ignore_index= True)
    data['therapies'] =  [[0] for r in range(len(data))]
    
    for i, row in data.iterrows():                  # for each row => each date of medical reports

        list_of_dates = []
        list_of_therapies = []
        dict_therap = dict()
        
        for therapy, date in zip(therapies, dates):
            if row[therapy] != 0:                       ## gather all the therapies into a dict with the id and the current date (date_report)

                dict_therap['Num_dossier'] = row['Num_dossier']

                dict_therap[therapy] = row[therapy]

                date_therapy = datetime.strptime(row[date], '%Y-%m-%d')
                dict_therap[date] = date_therapy

                Date_exam = datetime.strptime(row['Date_exam'], '%Y%m%d')
                dict_therap['date_event'] = Date_exam

                list_of_therapies.append(therapy) 
                list_of_dates.append(date_therapy)  

        list_of_therapies.append('Date_event')          ## gather the list of therapies and the dates for each row 
        list_of_dates.append(Date_exam)   


        sorted_dates = sorted(zip(list_of_dates, list_of_therapies))         ## sort the dates from the less recent to the most recent event 
        
        if sorted_dates[0][1] == 'Date_event':          
            ## If the less recent event between therapies and the current day, we don't have therapy yet = 0
            ## I need to see if within the BC window we will find the date_event (in a backward way)
 
            for j in range(len(sorted_dates)):  
                if sorted_dates[j][1] != 'Date_event':
                    window_dates = [sorted_dates[j][0]+timedelta(days=i) for i in range(-window, window+1)]
                   
                    if dict_therap['date_event'] in window_dates:
                        data.loc[i, 'therapies'].append(config[sorted_dates[j][1]])
                     #   data.at[i, 'therapies'] += int_to_list(config[sorted_dates[j][1]])
                      
    

        for j in range(len(sorted_dates)):  
            if sorted_dates[j][1] != 'Date_event':
                if timestamps == 'regular':
                    seq_of_dates = [sorted_dates[j][0]+timedelta(days=n*ts[sorted_dates[j][1]]) for n in range(number_of_cycles)]
                elif timestamps == 'irregular':
                    seq_of_dates = [date1]+[date1+timedelta(days=n) for n in ts[sorted_dates[j][1]]]

                if therap_window == True:
                    seq_of_dates = [seq_of_dates[i]+timedelta(days=j) for j in range(-window, window+1) for i in range(len(seq_of_dates))]

                if sorted_dates[j][1] == 'breast_surgery':
                    seq_of_dates = [sorted_dates[j][0]+timedelta(days=i) for i in range(-window, window+1)]


                if dict_therap['date_event'] in seq_of_dates:
                    data.loc[i, 'therapies'].append(config[sorted_dates[j][1]])
                   # data.at[i, 'therapies'] += int_to_list(config[sorted_dates[j][1]])
    
    ## return list with no duplicates
    data.loc[:,'therapies'] = data['therapies'].apply(lambda x: list(dict.fromkeys(x)))
    
    ## remove 0 when there is therapies in the list 
    
    data.loc[:,'therapies'] = data['therapies'].apply(lambda x:  x[1:] if len(x)>1 else x)
 
    return data 



    def add_delay(seq, delay):
        return seq+[delay]+['CLS']

#@jit(nopython=True, parallel=True)
def cal_subtherapies(df, fill_with = 'string'):
    
    data = df.copy()
    ## I use therapies in dict to have access to the code directly when using data 
    if fill_with == 'int':
        therapies = {'breast_surgery':1, 'ht':2, 'rt':3, 'ct':4, 'adj_antiher2':5, 'no therapie': 0}
        subtherapies = {therapies['breast_surgery']: {(2, 1): 1, (2, 2): 2, (2, 3): 3, (2, 0): 4, (1, 1): 5, (1, 2): 6, (1, 3): 7, (1, 0): 8}, 
                  therapies['ht']: {1: 9, 2: 10, 3: 11}, therapies['rt']: {1: 12}, therapies['ct']: {1:13, 2:14, 3: 15, 'nan' :16} , therapies['adj_antiher2']:{1: 17}
        }
        data['sub_therapies'] =  [[0] for r in range(len(data))]
        
        
    elif fill_with == 'string':
        therapies = {
        'breast_surgery':'chirurgie', 'ht':'hormone therapie', 'rt': 'radio_therapie', 'ct':'chimiotherapie', 'adj_antiher2':'adjuvant antiher2', 'no therapy': 'pas de therapie'
    }
            
        subtherapies = {therapies['breast_surgery']: {(2, 1): 'mastectomie et biopsie du ganglion sentinelle', 
                                                        (2, 2): 'mastectomie et dissection ganglionnaire axillaire', 
                                                        (2, 3): 'mastectomie et BGS + DSA', 
                                                        (2, 0): 'mastectomie seule', 
                                                        (1, 1): 'tumorectomie et biopsie du ganglion sentinelle', 
                                                        (1, 2): 'tumorectomie et dissection ganglionnaire axillaire', 
                                                        (1, 3): 'tumorectomie et BGS + DSA', 
                                                        (1, 0): 'tumorectomie seule'}, 
                  therapies['ht']: {1: 'tamoxifene', 
                                    2: 'inhibiteur aromatase', 
                                    3: 'autres'},
                          therapies['rt']: {1: 'radio_therapie'}, 
                          therapies['ct']: {1:'anthra-taxanes', 
                                            2:'anthra',
                                            3: 'taxanes', 
                                            4: 'autres',
                                            'nan' : 'non renseigné'} , 
                          therapies['adj_antiher2']:{1: 'antiher2 adjuvant'}
        }
        data['sub_therapies'] =  [['pas de subtherapie'] for r in range(len(data))]
    
    # breast surgery : (x, y) : y:(1: Sentinel Node Biopsy, 2:  Axillary node dissection, 3: Both SND and AND, 0: no axillar surgery)
    # x: (1: Lumpectomy, 2: Mastectomy)
    # ht :( 1: tamoxifen, 2: aromatase, 3: others)
    # ct : 1: anthra-taxanes, 2: anthra, 3: others, nan: missing 
    
    
    ## columns needed : 'axillary_surgery_4cl', 'breast_surgery', ht_type_3cl, 'adj_ct_regimen'
    cols = {therapies['breast_surgery']: ['breast_surgery_3cl', 'axillary_surgery_4cl'], therapies['ht']: 'ht_type_3cl', 
            therapies['rt']:'rt', therapies['ct']: 'adj_ct_regimen', therapies['adj_antiher2']:'adj_antiher2'}
    
   
    
    for i, row in data.iterrows():                  # for each row => each date of medical reports
        try: 
            if row['therapies'] != ['no therapie']:
                for j in row['therapies']: 
                    if j == therapies['breast_surgery']:  # if it's breast surgery
                        data.loc[i, 'sub_therapies'].append(subtherapies[j][(row[cols[j][0]],row[cols[j][1]])])
                        #data.at[i, 'sub_therapies'] += int_to_list(subtherapies[j][(row[cols[j][0]],row[cols[j][1]])])

                    else: 

                        if pd.isna(row[cols[j]]) :
                            data.loc[i, 'sub_therapies'].append(subtherapies[j]['nan'])
                        else:
                            data.loc[i, 'sub_therapies'].append(subtherapies[j][row[cols[j]]])
                        #data.at[i, 'sub_therapies'] += int_to_list(subtherapies[j][row[cols[j]]])

        except:
            data.loc[i, 'sub_therapies'].append('pas de sous therapie')
            
    ## return list with no duplicates
    data.loc[:,'sub_therapies'] = data['sub_therapies'].apply(lambda x: list(dict.fromkeys(x)))
    
    ## remove 0 when there is therapies in the list 
    
    data.loc[:,'sub_therapies'] = data['sub_therapies'].apply(lambda x:  x[1:] if len(x)>1 else x)
 
    return data 


class spark_init(object):
    def __init__(self, sparkConfig=None, name='Multimod_BEHRT'):
        self._setup_spark(sparkConfig)
        self.sc, self.sqlContext = self._init_spark(name=name)

    def _setup_spark(self, sparkConfig):

        if sparkConfig == None:
            config = {'memory': '300g', 'excutors': '4', 'exe_mem': '50G', 'result_size': '80g',
                      'temp': '/home/yikuan/tmp', 'offHeap':'16g'}
        else:
            config = sparkConfig

        os.environ["PYSPARK_PYTHON"] = "" # python spark path
        pyspark_submit_args = ' --driver-memory ' + config['memory'] + ' --num-executors ' + config['excutors'] + \
                              ' --executor-memory ' + config['exe_mem']+ \
                              ' --conf spark.driver.maxResultSize={} --conf spark.memory.offHeap.size={} --conf spark.local.dir={}'\
                                  .format(config['result_size'], config['offHeap'], config['temp']) +\
                              ' pyspark-shell'

        os.environ["PYSPARK_SUBMIT_ARGS"] = pyspark_submit_args

    def _init_spark(self, name='Multimod_BEHRT'):
        sc = pyspark.SparkContext(appName=name)
        sqlContext = SQLContext(sc)
        sqlContext.sql("SET spark.sql.parquet.binaryAsString=true")
        return sc, sqlContext
    
def read_txt(sc, sqlContext, path):
    """read from txt to pyspark dataframe"""
    file = sc.textFile(path)
    head = file.first()
    content = file.filter(lambda line: line != head).map(lambda k: k.split('\t'))
    df = sqlContext.createDataFrame(content, schema=head.split('\t'))
    return df


def read_parquet(sqlContext, path):
    """read from parquet to pyspark dataframe"""
    return sqlContext.read.parquet(path)

def read_csv(sqlContext, path):
    """read from parquet to pyspark dataframe"""
    return sqlContext.read.csv(path, header=True)

def int_to_str(x):
    return [str(int(f)) for f in x]

def float_to_str(list_of_int):
    float_to_int = [ int(x) for x in list_of_int]
    return [str(x) for x in float_to_int]

def flatten_list(x):
    try:
        res = [item for sublist in x for item in sublist]
    except:
        res = x
    return res

def flatten_array(x):
    x = np.hstack(x)
    x = x.flatten()
    return x



def cal_therapies_fr(df, therap_window = True, window = 14):

    """ This function makes the therapies sequential using the date they've been administrated to display their code
    surgery : 1, 'ht':2, 'rt':3, 'ct':4, 'antiher2':5 
    data : the data with the therapies in binary form
    number_of_cycles: number of cycles of cures for treatments  # cf. 'nb_cycles_adj_ct'
    timestamps: The regularity of therapy session, each 5days: regular vs irregular
    ts: the number of days between two treatments or the sequence of days when it's irregular eq:[2, 5, 7]: 2, 5 and 7 days after the first treatment
    # Take into account the date_end of the treatment in the sequence ts 
    window : The number of days to add before and after the treatment in order to be matched with reports writen before and after the treatment"""
    df = data.copy()
    
    therapies = ['breast_surgery', 'ht', 'rt', 'ct', 'adj_antiher2']
    dates = ['dat_first_surg', 'dat_first_ht', 'dat_first_rt', 'dat_first_ct', 'dat_first_antiher2']
    config = {
        'breast_surgery':1, 'ht':2, 'rt':3, 'ct':4, 'adj_antiher2':5, 'no therapy': 0
    }
    
    ts = {'breast_surgery':0, 'ct': 6, 'rt': 6, 'ht': 6, 'adj_antiher2':6}
    # The timestamps between cures: 
    
    data = df.copy()
    data['therapies'] =  [[0] for r in range(len(data))]
    
    for i, row in data.iterrows():                  # for each row => each date of medical reports
        
     #   print("... row "+str(i)+" ...")
       # print(data.loc[i, 'therapies'])
        list_of_dates = []
        list_of_therapies = []
        dict_therap = dict()
        
        #data.at[i, 'therapies'] = data.apply(lambda x:[], axis=1)
        for therapy, date in zip(therapies, dates):
            if row[therapy] != 0:                       ## gather all the therapies into a dict with the id and the current date (date_report)
                try: 
                    dict_therap['Num_dossier'] = row['Num_dossier']

                    dict_therap[therapy] = row[therapy]

                    date_therapy = datetime.strptime(row[date], '%Y-%m-%d')
                    dict_therap[date] = date_therapy

                    Date_exam = datetime.strptime(row['Date_exam'], '%Y%m%d')
                    dict_therap['date_event'] = Date_exam

                    list_of_therapies.append(therapy) 
                    list_of_dates.append(date_therapy)  
               
                except:
                    pass

        list_of_therapies.append('Date_event')          ## gather the list of therapies and the dates for each row 
        list_of_dates.append(Date_exam)   


        sorted_dates = sorted(zip(list_of_dates, list_of_therapies))         ## sort the dates from the less recent to the most recent event 
  
        if sorted_dates[0][1] == 'Date_event':          
            ## If the less recent event between therapies and the current day, we don't have therapy yet = 0
            ## I need to see if within the BC window we will find the date_event (in a backward way)
 
            for j in range(len(sorted_dates)):  
                if sorted_dates[j][1] != 'Date_event':
                    window_dates = [sorted_dates[j][0]+timedelta(days=i) for i in range(-window, window+1)]
                   
                    if dict_therap['date_event'] in window_dates:
                       # data.loc[i, 'therapies'].append(config[sorted_dates[j][1]])
                        data.at[i, 'therapies'] += int_to_list(config[sorted_dates[j][1]])
                      
                  #  else:
                     #   data.at[i, 'therapies'] = int_to_list(0)
               
      
        for j in range(len(sorted_dates)):  
            if sorted_dates[j][1] != 'Date_event':
                
                if sorted_dates[j][1] == 'breast_surgery':
                    seq_of_dates = [sorted_dates[j][0]+timedelta(days=i) for i in range(-window, window+1)]

                
                if sorted_dates[j][1] == 'ht':
               
                    # For ht it will be 1 injection per day for 5, 7-10 years for patients with lymph nodes affected
                    if row['pnuicc_4cl'] == 1 or pd.isna(row['pnuicc_4cl']) == True:  # if 0 nodes involved or missing information (520patients)
                        
                        seq_of_dates =  pd.date_range(start = sorted_dates[j][0], end = sorted_dates[j][0]+timedelta(days=5*365.2425))
                    else:  
                        seq_of_dates =  pd.date_range(start = sorted_dates[j][0], end= sorted_dates[j][0]+timedelta(days=8.5*365.2425))
                
                if sorted_dates[j][1] == 'rt':
               
                    # 1 cure per day for 4-6 weeks (5 is choosen)
                    seq_of_dates =  pd.date_range(start = sorted_dates[j][0], end= sorted_dates[j][0]+timedelta(weeks=5))
                
                if sorted_dates[j][1] == 'adj_antiher2':
             
                    # 18 injections each 3 weeks (combined with CT) (verifie ca)
                    seq_of_dates = [sorted_dates[j][0]+timedelta(days=n*21) for n in range(18)]
                
                if sorted_dates[j][1] == 'ct':
                   
                    if row['adj_ct_regimen'] == 1 or row['adj_ct_regimen'] == 2  :
                        delay = sorted_dates[j][0]
                        seq_of_dates = []
                        try:
                            for i in range(row['nb_cycles_adj_ct']):
                                seq_of_dates = seq_of_dates+[delay+timedelta(days=n*21) for n in range(4)]
                                delay = seq_of_dates[-1]+timedelta(weeks=3+3+11)  # CF TIMELINE => 3weeks after we start the taxol cure, then 1cure per week for 11 weeks 
                                                                                    # and 3 more weeks between cycles (21d + 77d + 21d)
                        except:
                            seq_of_dates = [sorted_dates[j][0]+timedelta(days=n*21) for n in range(4)]
                        
                    if row['adj_ct_regimen'] == 1:
                        delay = sorted_dates[j][0]
                        seq_of_dates = []
                        try:
                            for i in range(row['nb_cycles_adj_ct']):
                                seq_of_dates = seq_of_dates+[delay+timedelta(days=n*7) for n in range(11)]
                                delay = seq_of_dates[-1]+timedelta(days=21)
                        except:
                            seq_of_dates = [sorted_dates[j][0]+timedelta(days=n*7) for n in range(11)]
                
                 ### Problem: I don't have the information regarding the molecule used: Let's assume that taxotere or taxol is used
                 ## 
                 #   if molecules == 'taxol':
                 #       seq_of_dates = [sorted_dates[j][0]+timedelta(days=n*7) for n in range(11)]
                 #   if molecules == 'taxotère':
                 #       seq_of_dates = [sorted_dates[j][0]+timedelta(days=n*21) for n in range(4)]      
                
                if therap_window == True:
                    seq_of_dates = [seq_of_dates[i]+timedelta(days=j) for j in range(-window, window+1) for i in range(len(seq_of_dates))]

                if dict_therap['date_event'] in seq_of_dates:
                    data.at[i, 'therapies'] += int_to_list(config[sorted_dates[j][1]])
               
                    
    ## return list with no duplicates
    data.loc[:,'therapies'] = data['therapies'].apply(lambda x: list(dict.fromkeys(x)))
    
    ## remove 0 when there is therapies in the list 
    
    data.loc[:,'therapies'] = data['therapies'].apply(lambda x:  x[1:] if len(x)>1 else x)
 
    return data 

def divide_frame(data, n_patient, n):
    """ this function allows us to split the huge dataframe of reports into small ones."""
    
    list_of_frames = []
    list_of_chunks = list(divide_chunks(n_patient, n))
    grouped = data.groupby(data.Num_dossier)
    
    for i in list_of_chunks:
        list_of_frames.append(pd.concat([grouped.get_group(name) for name in i]))
    
    return list_of_frames


def divide_chunks(l, n):
    """This function splits the list of patients into list of n patients that will be used as patients for each subgroup of dataframe"""
      
    # looping till length l
    for i in range(0, len(l), n): 
        yield l[i:i + n]


def cal_therapies(df, therap_window = True, window = 14, fill_with = 'string'):

    """ This function makes the therapies sequential using the date they've been administrated to display their code
    surgery : 1, 'ht':2, 'rt':3, 'ct':4, 'antiher2':5 
    data : the data with the therapies in binary form
    number_of_cycles: number of cycles of cures for treatments  # cf. 'nb_cycles_adj_ct'
    timestamps: The regularity of therapy session, each 5days: regular vs irregular
    ts: the number of days between two treatments or the sequence of days when it's irregular eq:[2, 5, 7]: 2, 5 and 7 days after the first treatment
    # Take into account the date_end of the treatment in the sequence ts 
    window : The number of days to add before and after the treatment in order to be matched with reports writen before and after the treatment"""

    data = df.copy()
    therapies = ['breast_surgery', 'ht', 'rt', 'ct', 'adj_antiher2']
    dates = ['dat_first_surg', 'dat_first_ht', 'dat_first_rt', 'dat_first_ct', 'dat_first_antiher2']
    
    if fill_with == 'int':
        config = {
            'breast_surgery':1, 'ht':2, 'rt':3, 'ct':4, 'adj_antiher2':5, 'no therapy': 0
        }
        data['therapies'] =  [[0] for r in range(len(data))]
        
    elif fill_with == 'string':
        config = {'breast_surgery':'chirurgie', 'ht':'hormone therapie', 'rt':'radio_therapie', 'ct':'chimiotherapie', 'adj_antiher2':'adjuvant antiher2', 'no therapy': 'pas de therapie'}
        data['therapies'] =  [['pas de therapie'] for r in range(len(data))]
        
  
    for i, row in data.iterrows():                  # for each row => each date of medical reports
        
        list_of_dates = []
        list_of_therapies = []
        dict_therap = dict()
       # print(row['
        #data.at[i, 'therapies'] = data.apply(lambda x:[], axis=1)
        for therapy, date in zip(therapies, dates):
          
            if row[therapy] != 0:                       ## gather all the therapies into a dict with the id and the current date (date_report)
                try:
                    dict_therap['Num_dossier'] = row['Num_dossier']

                    dict_therap[therapy] = row[therapy]
                    try: 
                        date_therapy = datetime.strptime(row[date], '%Y%m%d')
                    except: 
                        date_therapy = datetime.strptime(row[date], '%Y-%m-%d')
                    dict_therap[date] = date_therapy
                  #  print(row['Date_exam'])
                    Date_exam = datetime.strptime(row['Date_exam'], '%Y%m%d')
                    dict_therap['date_event'] = Date_exam

                    list_of_therapies.append(therapy) 
                    list_of_dates.append(date_therapy)  
               
                except:
                    
                    pass

        list_of_therapies.append('Date_event')          ## gather the list of therapies and the dates for each row 
        list_of_dates.append(Date_exam)   

    
        if sorted_dates[0][1] == 'Date_event':          
            ## If the less recent event between therapies and the current day, we don't have therapy yet = 0
            ## I need to see if within the BC window we will find the date_event (in a backward way)
 
            for j in range(len(sorted_dates)):  
                if sorted_dates[j][1] != 'Date_event':
                    window_dates = [sorted_dates[j][0]+timedelta(days=i) for i in range(-window, window+1)]
                   
                    if dict_therap['date_event'] in window_dates:
                       # data.loc[i, 'therapies'].append(config[sorted_dates[j][1]])
                        data.at[i, 'therapies'] += int_to_list(config[sorted_dates[j][1]])
                      
                  #  else:
                     #   data.at[i, 'therapies'] = int_to_list(0)
               
      
        for j in range(len(sorted_dates)):  
            if sorted_dates[j][1] != 'Date_event':
                
                if sorted_dates[j][1] == 'breast_surgery':
                    seq_of_dates = [sorted_dates[j][0]+timedelta(days=i) for i in range(-window, window+1)]

                
                if sorted_dates[j][1] == 'ht':
               
                    # For ht it will be 1 injection per day for 5, 7-10 years for patients with lymph nodes affected
                    if row['pnuicc_4cl'] == 1 or pd.isna(row['pnuicc_4cl']) == True:  # if 0 nodes involved or missing information (520patients)
                        
                        seq_of_dates =  pd.date_range(start = sorted_dates[j][0], end = sorted_dates[j][0]+timedelta(days=5*365.2425))
                    else:  
                        seq_of_dates =  pd.date_range(start = sorted_dates[j][0], end= sorted_dates[j][0]+timedelta(days=8.5*365.2425))
                
                if sorted_dates[j][1] == 'rt':
               
                    # 1 cure per day for 4-6 weeks (5 is choosen)
                    seq_of_dates =  pd.date_range(start = sorted_dates[j][0], end= sorted_dates[j][0]+timedelta(weeks=5))
                
                if sorted_dates[j][1] == 'adj_antiher2':
             
                    # 18 injections each 3 weeks (combined with CT) (verifie ca)
                    seq_of_dates = [sorted_dates[j][0]+timedelta(days=n*21) for n in range(18)]
                
                if sorted_dates[j][1] == 'ct':
                   
                    if row['adj_ct_regimen'] == 1 or row['adj_ct_regimen'] == 2  :
                        delay = sorted_dates[j][0]
                        seq_of_dates = []
                        try:
                            for i in range(row['nb_cycles_adj_ct']):
                                seq_of_dates = seq_of_dates+[delay+timedelta(days=n*21) for n in range(4)]
                                delay = seq_of_dates[-1]+timedelta(weeks=3+3+11)  # CF TIMELINE => 3weeks after we start the taxol cure, then 1cure per week for 11 weeks 
                                                                                    # and 3 more weeks between cycles (21d + 77d + 21d)
                        except:
                            seq_of_dates = [sorted_dates[j][0]+timedelta(days=n*21) for n in range(4)]
                        
                    if row['adj_ct_regimen'] == 1:
                        delay = sorted_dates[j][0]
                        seq_of_dates = []
                        try:
                            for i in range(row['nb_cycles_adj_ct']):
                                seq_of_dates = seq_of_dates+[delay+timedelta(days=n*7) for n in range(11)]
                                delay = seq_of_dates[-1]+timedelta(days=21)
                        except:
                            seq_of_dates = [sorted_dates[j][0]+timedelta(days=n*7) for n in range(11)]
                
                 ### Problem: I don't have the information regarding the molecule used: Let's assume that taxotere or taxol is used
                 ## 
                 #   if molecules == 'taxol':
                 #       seq_of_dates = [sorted_dates[j][0]+timedelta(days=n*7) for n in range(11)]
                 #   if molecules == 'taxotère':
                 #       seq_of_dates = [sorted_dates[j][0]+timedelta(days=n*21) for n in range(4)]      
                
                if therap_window == True:
                    seq_of_dates = [seq_of_dates[i]+timedelta(days=j) for j in range(-window, window+1) for i in range(len(seq_of_dates))]
    
                
                if dict_therap['date_event'] in seq_of_dates:
                    data.at[i, 'therapies'] += int_to_list(config[sorted_dates[j][1]])
                 
               
                    
    ## return list with no duplicates
    data.loc[:,'therapies'] = data['therapies'].apply(lambda x: list(dict.fromkeys(x)))
    
    ## remove 0 when there is therapies in the list 
    
    data.loc[:,'therapies'] = data['therapies'].apply(lambda x:  x[1:] if len(x)>1 else x)
 
    return data 


def clean_accents(df):
    for i, row in df.iterrows(): 

        for inputs in ['inputs_quantiles', 'inputs_normal_range', 'inputs_curve_intersections',
       'inputs_bis_quantiles', 'inputs_bis_normal_range', 'inputs_bis_curve_intersections'] :
            row[inputs] = [i.replace('Ã©', 'é') for i in row[inputs]] 
            row[inputs] = [i.replace('Ã¨', 'è') for i in row[inputs]] 
            row[inputs] = [i.replace('Ã´', 'o') for i in row[inputs]] 
            row[inputs] = [i.replace('ô', 'o') for i in row[inputs]] 
            
            
    for col in ['age', 'delays']:
        df[col] = df[col].apply(lambda x: float_to_str(x))
   
    return df


def get_index_positions(list_of_elems, element):
    ''' Returns the indexes of all occurrences of give element in
    the list- listOfElements '''
    index_pos_list = []
    index_pos = 0
    while True:
        try:
            # Search for item in list from indexPos to the end of list
            index_pos = list_of_elems.index(element, index_pos)
            # Add the index position in list
            index_pos_list.append(index_pos)
            index_pos += 1
        except ValueError as e:
            break
    return index_pos_list

def add_sep(list_to_change, element= 'CLS', change_with = 'SEP'):
    
    indexes = get_index_positions(list_to_change, element)
    
    for i, j in zip(indexes[1:], range(len(indexes)-1)):
        list_to_change.insert(i+j, change_with)
        
    return list_to_change



def categorize_delays(delay):
    cat_delay = dict()
    for week in range(4):
        cat_delay[week] = 'W'+str(week)
    for month in range(1,13):
        cat_delay[int(np.round(month*4.3482))] = 'M'+str(month)
    ## for delay more thana year 
    cat_delay[1000] = 'LT' # Long term

    if delay in cat_delay.keys():
        return cat_delay[delay]
    elif delay <= 52: 
        keys_array = np.array(list(cat_delay.keys()))
        return cat_delay[keys_array[keys_array>delay][0]]
    else: 
        return cat_delay[1000]
        

def set_grade(row):
    if pd.isna(row):
        return 'Missing'
    elif row == 'Non précisé':
        return 'Missing'
    else:
        return row

def set_nodes(x, y):
    if pd.isna(x) & pd.isna(y):
        return 'Missing'
    elif pd.isna(x):
        return y
    elif pd.isna(y):
        return x
    elif pd.isna(x) == False & pd.isna(y) == False:
        return x+y
    
      
def set_modes(data, var_1, var_2):
    df = (data.groupby(var_1)[var_2]
               .apply(lambda x: x.mode().iat[0])
               .reset_index())
    return df

def set_size(data, x, how='clinical'):
    if how == 'clinical':
        df = set_modes(data, 'TUICC', 'TCLIN')
        if pd.isna(x.TCLIN) == False:
            return x.TCLIN
        elif pd.isna(x.TUICC) == False: 
            return df.loc[df['TUICC']==x.TUICC]['TCLIN'].values[0]
        else:
            return 'Missing'
    
    elif how == 'pathological':
        df = set_modes(data, 'PTUICC', 'TINF')
        if pd.isna(x.TINF) == False:
            return x.TINF
        elif pd.isna(x.PTUICC) == False:
            return df.loc[df['PTUICC']==x.PTUICC]['TINF'].values[0]
        else:
            return 'Missing'
        
def set_tumor_size(row):
    if row.clinical_ts != 'Missing':
        if row.Date_exam < row.dat_first_surg:
            return row.clinical_ts/10 # en cm 
        else:
            if row.pathological_ts != 'Missing':
                return row.pathological_ts/10 #en cm 
            else:
                return 'Missing'
    else:
        return row.clinical_ts/10
        
def set_subtype(row):
    if pd.isna(row) == False:
        return row
    else:
        return 'Missing'    
    
def categorize_size(size):
    
    bins = list(np.arange(0.0, 30.0, .5))
    labels = list()
    for i in bins[1:]:
        labels.append('S'+str(i))
    return pd.cut([size], bins=bins, labels=labels, include_lowest=True)
  
def categorize_nodes_num(nodes):
    bins = list(range(-1, 50))
    labels = list()
    for i in bins[1:]:
        labels.append('N'+str(i))
  
    return pd.cut([nodes], bins=bins, labels=labels, include_lowest=True)
    
def categorize_grade(grade):
    if grade == 'Non évaluable':
        return 'GX'
    elif grade != 'Missing':
        return 'G'+str(grade)
    else:
        return grade
    
def categorize_st(st):
    if st == 1:
        return 'Luminal'
    elif st == 2:
        return 'TNBC'
    elif st == 3:
        return 'HER2+/RH+'
    elif st == 4:
        return 'HER2+/RH-'
    
    
from joblib import Parallel, delayed
from tqdm import tqdm

def set_window(date, window):
    return [date+timedelta(days=j) for j in range(-window, window+1)]


def cal_subtherapies(data, window=14):

    therapies = ['chirurgie', 'ht', 'rt', 'ct', 'antiher2']
    dates = [['dat_first_surgery', 'dat_second_surgery', 'dat_third_surgery'], ['DATDHT.q1','DATFHT.q1','DATDHT.q2','DATFHT.q2','DATDHT.q3','DATFHT.q3','DATDHT.q4','DATFHT.q4'], 
             ['DATDRT', 'DATFRT'], 
                ['DATDCT.f1', 'DATFCT.f1', 'DATDCT.f2', 'DATFCT.f2'], 
                ['dat_beg_first_antiher2', 'dat_end_first_antiher2', 'dat_beg_second_antiher2', 'dat_end_second_antiher2']]


    subtherapies = {'chirurgie' :['surg_first_act_completed', 'surg_second_act_completed', 'surg_third_act_completed'],
                   'ht': ['TYPHT.q1_v2','TYPHT.q2_v2','TYPHT.q3_v2','TYPHT.q4_v2'],
                   'rt': ['RT_subtype'], 'ct': ['typct_1e_instance', 'typct_2e_instance'],
                    'antiher2': ['antiher2_molecule_first_instance', 'antiher2_molecule_second_instance']}

    data['sub_therapies'] =  [['pas de sous therapie'] for r in range(len(data))]


    date_therapy = list()

    for index, row in data.iterrows(): 
        list_of_dates = []
        list_of_therapies = []

        dict_therap = dict()

        for therapy, date in zip(therapies, dates):
            if row[therapy] != 0:

                dict_therap['Num_dossier'] = row['Num_dossier']
                dict_therap[therapy]= row[therapy]

                #for d in date: 
              #  date_therapy.append(str_to_date(row[date]))
                dict_therap[tuple(date)] = [str_to_date(row[d]) for d in date]


                Date_exam = str_to_date(row['Date_exam'])
                dict_therap['Date_event'] = Date_exam


                list_of_therapies.append(therapy)
                list_of_dates.append([str_to_date(row[d]) for d in date])

        list_of_therapies.append('Date_event')
        list_of_dates.append([Date_exam])

        sorted_dates = sorted(zip(list_of_dates, list_of_therapies))
      #  print(sorted_dates)
       # print()
        if sorted_dates[0][1] == 'Date_event':
             ## If the less recent event between therapies and the current day, we don't have therapy yet = 0
        ## I need to see if within the BC window we will find the date_event (in a backward way)

            for j in range(len(sorted_dates)):

                if sorted_dates[j][1] != 'Date_event':
                    window_dates = list()
                   ## We assume that reports that happened 14 days before and 14 days after the date of the therapy (beg and end for certain therapies) are somehow concern therapies

                    for d in sorted_dates[j][0]:
                        window_dates.append([d+timedelta(days=i) for i in range(-window, window+1) if pd.isna(d)==False])

                    step = 0
                    while step < len(window_dates):
                        if dict_therap['Date_event'] in window_dates[step]:
                            data.at[index, 'sub_therapies'] += row[subtherapies[sorted_dates[j][1]][step]]
                        step +=1

        for j in range(len(sorted_dates)):
            if sorted_dates[j][1] != 'Date_event':

                if sorted_dates[j][1] == 'chirurgie':
                    ## For each surgey, we use 14 days before and 14 days to display information about surgery on the sequence

                    step=0
                    while step < len(sorted_dates[j][0]):

                        seq_of_dates = []
                        seq_of_dates = [sorted_dates[j][0][step]+timedelta(days=i) for i in range(-window, window+1) if pd.isna(sorted_dates[j][0][step]) == False]

                        if dict_therap['Date_event'] in seq_of_dates:

                          #  print(dict_therap['Date_event'])
                         #   print(seq_of_dates)
                         #   print(index)

                            data.at[index, 'sub_therapies'] += row[subtherapies[sorted_dates[j][1]][step]]

                        step+=1

                if sorted_dates[j][1] == 'ht':
                    ## Normally for ht, we have 1 injection per day for 5, 7-10 years for patients with lymph nodes affected (from the therapeutic protocol). We will use it if the information number of doses is unavailable. 
                    seq_of_dates = []
                    date_ht_index = 0
                    step = 0
                    while date_ht_index <= len(sorted_dates[j][0])-1:
                       # if pd.isna(sorted_dates[j][0][date_ht_index]) == False:

                        if pd.isna(sorted_dates[j][0][date_ht_index+1]) == False:
                            seq_of_dates = list(pd.date_range(start = sorted_dates[j][0][date_ht_index], end=sorted_dates[j][0][date_ht_index+1]))

                        else: 
                            if pd.isna(sorted_dates[j][0][date_ht_index]) == True:
                                pass

                            else:

                                if row['pnuicc_4cl'] == 1 or pd.isna(row['pnuicc_4cl']) == True:  # if 0 nodes involved or missing information (520patients)
                                    seq_of_dates = list(pd.date_range(start = sorted_dates[j][0][date_ht_index], end = sorted_dates[j][0][date_ht_index]+timedelta(days=5*365.2425)))
                                else: 
                                    seq_of_dates = list(pd.date_range(start = sorted_dates[j][0][date_ht_index], end= sorted_dates[j][0][date_ht_index]+timedelta(days=8.5*365.2425)))


                        seq_of_dates = seq_of_dates + [seq_of_dates[0]+timedelta(days=j) for j in range(-7, 7+1)]+[seq_of_dates[-1]+timedelta(days=j) for j in range(-7, 7+1)]
                      #  print(seq_of_dates[-10:])
                      #  print(date_ht_index)
                    #    print()

                        if dict_therap['Date_event'] in seq_of_dates:
                            if pd.isna(row[subtherapies[sorted_dates[j][1]][step]]): ## when I can have many molec at the same time make sure I don't add nan 
                                pass
                            else:
                                data.at[index, 'sub_therapies'] += [row[subtherapies[sorted_dates[j][1]][step]]]

                        date_ht_index+=2
                        step+=1

                if sorted_dates[j][1] == 'rt':
                    seq_of_dates = []
                        ## we will use number of doses of rt. 
                    if pd.isna(row['DOSBOOSTRT']) == False:
                        seq_of_dates = pd.date_range(start = sorted_dates[j][0][0], periods = row['DOSBOOSTRT'])
                    elif pd.isna(row['DOSGPRT']) == False:
                        seq_of_dates = pd.date_range(start = sorted_dates[j][0][0], periods = row['DOSGPRT'])
                    else:
                        # 1 cure per day for 4-6 weeks (5 is choosen)
                        if pd.isna(sorted_dates[j][0][1]) == True:
                            seq_of_dates = pd.date_range(start = sorted_dates[j][0][0], end= sorted_dates[j][0][0]+timedelta(weeks=5))
                        else:
                            seq_of_dates = pd.date_range(start = sorted_dates[j][0][0], end=sorted_dates[j][0][1])

                    seq_of_dates = [seq_of_dates[i]+timedelta(days=j) for j in range(-7, 7+1) for i in range(len(seq_of_dates))]

                    if dict_therap['Date_event'] in seq_of_dates:
                        data.at[index, 'sub_therapies'] += row[subtherapies[sorted_dates[j][1]]][0]


                if sorted_dates[j][1] == 'antiher2':
                    date_antiher2_index = 0
                    step = 0
                    seq_of_dates = []

                    while date_antiher2_index <= len(sorted_dates[j][0])-1:
                        if pd.isna(sorted_dates[j][0][date_antiher2_index+1]) == False:
                            # if date_fin is available compute number of cures, num_days/21. Check if the cure is done each 3 weeks.

                            num_cures = (str_to_date(sorted_dates[j][0][date_antiher2_index+1]) - str_to_date(sorted_dates[j][0][date_antiher2_index])).days/21
                            num_cures = int(np.ceil(num_cures))
                            seq_of_dates = ([sorted_dates[j][0][date_antiher2_index]+timedelta(days=n*21) for n in range(num_cures)])

                        else:
                            if pd.isna(sorted_dates[j][0][date_antiher2_index]) == True:
                                pass

                            else:
                            # if no date_end is specified, take 18 injections
                            # 18 injections each 3 weeks (combined with CT) (verifie ca)
                                seq_of_dates = [sorted_dates[j][0][date_antiher2_index]+timedelta(days=n*21) for n in range(18)]

                        seq_of_dates = [seq_of_dates[i]+timedelta(days=j) for j in range(-7, 7+1) for i in range(len(seq_of_dates))]

                        if dict_therap['Date_event'] in seq_of_dates:
                            if pd.isna(row[subtherapies[sorted_dates[j][1]][step]]):
                                pass
                            else:
                                data.at[index, 'sub_therapies'] += row[subtherapies[sorted_dates[j][1]][step]]

                        date_antiher2_index+=2
                        step+=1



                if sorted_dates[j][1] == 'ct':
                        seq_of_dates = []
                        date_ct_index = 0
                        step = 0
                        while date_ct_index <= len(sorted_dates[j][0]) - 1:
                            periods = 21 #CT each 21days
                            if pd.isna(sorted_dates[j][0][date_ct_index]) == True:
                                pass

                            else:
                              #  if pd.isna(sorted_dates[j][0][date_ct_index+1]) == False:
                                if date_ct_index == 0:
                                    num_cycles = np.max([row['NBCYCT.q1.f1'], row['NBCYCT.q2.f1'], row['NBCYCT.q3.f1'], row['NBCYCT.q4.f1'], row['NBCYCT.q5.f1']])

                                    if pd.isna(num_cycles):
                                        if 'TAXOL' in [row['NBCYCT.q1.f1'], row['NBCYCT.q2.f1'], row['NBCYCT.q3.f1'], row['NBCYCT.q4.f1'], row['NBCYCT.q5.f1']] :
                                            num_cycles = 11 
                                            periods = 7

                                        else: 
                                            num_cycles = 4

                                elif date_ct_index == 2:
                                    num_cycles = np.max([row['NBCYCT.q1.f2'], row['NBCYCT.q2.f2'],row['NBCYCT.q3.f2'], row['NBCYCT.q4.f2'],row['NBCYCT.q5.f2']])
                                    if pd.isna(num_cycles):
                                        if 'TAXOL' in [row['NBCYCT.q1.f1'], row['NBCYCT.q2.f1'], row['NBCYCT.q3.f1'], row['NBCYCT.q4.f1'], row['NBCYCT.q5.f1']] :
                                            num_cycles = 11 
                                        else: 
                                            num_cycles = 4

                                seq_of_dates =  seq_of_dates + [sorted_dates[j][0][date_ct_index]+timedelta(days=n*periods) for n in range(num_cycles)]
                                # knowing that CT last about 3 to 5 days we take 3days a window for each CT treatment 
                                seq_of_dates = seq_of_dates = [seq_of_dates[i]+timedelta(days=j) for j in range(-3, 3+1) for i in range(len(seq_of_dates))]

                            if dict_therap['Date_event'] in seq_of_dates: 
                                data.at[index, 'sub_therapies'] += row[subtherapies[sorted_dates[j][1]][step]]

                            date_ct_index+=2
                            step+=1


    ## return list with no duplicates
    data.loc[:,'sub_therapies'] = data['sub_therapies'].apply(lambda x: list(dict.fromkeys(x)))

    ## remove 0 when there is therapies in the list 

    data.loc[:,'sub_therapies'] = data['sub_therapies'].apply(lambda x:  x[1:] if len(x)>1 else x)
                                   
    return data


def cal_therapies(df, window=14):
                                   
    """ This function makes the therapies sequential using the date they've been administrated to display their code
    surgery : 1, 'ht':2, 'rt':3, 'ct':4, 'antiher2':5 
    data : the data with the therapies in binary form
    number_of_cycles: number of cycles of cures for treatments  # cf. 'nb_cycles_adj_ct'
    timestamps: The regularity of therapy session, each 5days: regular vs irregular
    ts: the number of days between two treatments or the sequence of days when it's irregular eq:[2, 5, 7]: 2, 5 and 7 days after the first treatment
    # Take into account the date_end of the treatment in the sequence ts 
    window : The number of days to add before and after the treatment in order to be matched with reports writen before and after the treatment"""
        
    data = df.copy()
    therapies = ['chirurgie', 'ht', 'rt', 'ct', 'antiher2']
    dates = [['dat_first_surgery', 'dat_second_surgery', 'dat_third_surgery'], ['DATDHT.q1','DATFHT.q1','DATDHT.q2','DATFHT.q2','DATDHT.q3','DATFHT.q3','DATDHT.q4','DATFHT.q4'], 
             ['DATDRT', 'DATFRT'], 
                ['DATDCT.f1', 'DATFCT.f1', 'DATDCT.f2', 'DATFCT.f2'], 
                ['dat_beg_first_antiher2', 'dat_end_first_antiher2', 'dat_beg_second_antiher2', 'dat_end_second_antiher2']]

    config = {'chirurgie':'chirurgie', 'ht':'hormone therapie', 'rt':'radiotherapie', 'ct':'chimiotherapie', 'antiher2':'adjuvant anti-her2', 'no therapy': 'pas de therapie'}
    
    data['therapies'] =  [['pas de therapie'] for r in range(len(data))]
    #data["therapies"] = [[] for r in range(len(data))]
    date_therapy = list()

    for index, row in data.iterrows(): 
        list_of_dates = []
        list_of_therapies = []

        dict_therap = dict()

        for therapy, date in zip(therapies, dates):
           
            if row[therapy] != 0:

                dict_therap['Num_dossier'] = row['Num_dossier']
                dict_therap[therapy]= row[therapy]

                #for d in date: 
              #  date_therapy.append(str_to_date(row[date]))
                dict_therap[tuple(date)] = [str_to_date(row[d]) for d in date]


                Date_exam = str_to_date(row['Date_exam'])
                dict_therap['Date_event'] = Date_exam


                list_of_therapies.append(therapy)
                list_of_dates.append([str_to_date(row[d]) for d in date])

        list_of_therapies.append('Date_event')
        list_of_dates.append([Date_exam])

        sorted_dates = sorted(zip(list_of_dates, list_of_therapies))
      #  print(sorted_dates)
        if sorted_dates[0][1] == 'Date_event':

             ## If the less recent event between therapies and the current day, we don't have therapy yet = 0
        ## I need to see if within the BC window we will find the date_event (in a backward way)

            for j in range(len(sorted_dates)):

                if sorted_dates[j][1] != 'Date_event':
                    window_dates = list()
                   ## We assume that reports that happened 14 days before and 14 days after the date of the therapy (beg and end for certain therapies) are somehow concern therapies

                    for d in sorted_dates[j][0]:
                        window_dates = window_dates + [d+timedelta(days=i) for i in range(-window, window+1) if pd.isna(d)==False]

                    if dict_therap['Date_event'] in window_dates:
                        data.at[index, 'therapies'] += int_to_list(config[sorted_dates[j][1]])
                      #  print(int_to_list(config[sorted_dates[j][1]]))
                    

        for j in range(len(sorted_dates)):
            if sorted_dates[j][1] != 'Date_event':

                if sorted_dates[j][1] == 'chirurgie':
                    ## For each surgey, we use 14 days before and 14 days to display information about surgery on the sequence
                    seq_of_dates = []

                    for date_chir in sorted_dates[j][0]:
                        seq_of_dates = seq_of_dates + [date_chir+timedelta(days=i) for i in range(-window, window+1) if pd.isna(date_chir) == False]

                   # seq_of_dates = [seq_of_dates[i]+timedelta(days=j) for j in range(-window, window+1) for i in range(len(seq_of_dates))]


                if sorted_dates[j][1] == 'ht':
                    ## Normally for ht, we have 1 injection per day for 5, 7-10 years for patients with lymph nodes affected (from the therapeutic protocol). We will use it if the information number of doses is unavailable. 
                    seq_of_dates = []

                    date_ht_index = 0
                    while date_ht_index <= len(sorted_dates[j][0])-1:

                        if pd.isna(sorted_dates[j][0][date_ht_index+1]) == False:
                            seq_of_dates = seq_of_dates + list(pd.date_range(start = sorted_dates[j][0][date_ht_index], end=sorted_dates[j][0][date_ht_index+1]))

                        else: 
                            if pd.isna(sorted_dates[j][0][date_ht_index]) == True:
                                pass

                            else:

                                if row['pnuicc_4cl'] == 1 or pd.isna(row['pnuicc_4cl']) == True:  # if 0 nodes involved or missing information (520patients)
                                    seq_of_dates = seq_of_dates + list(pd.date_range(start = sorted_dates[j][0][date_ht_index], end = sorted_dates[j][0][date_ht_index]+timedelta(days=5*365.2425)))
                                else: 
                                    seq_of_dates = seq_of_dates + list(pd.date_range(start = sorted_dates[j][0][date_ht_index], end= sorted_dates[j][0][date_ht_index]+timedelta(days=8.5*365.2425)))

                        date_ht_index+=2

                    seq_of_dates = seq_of_dates + [seq_of_dates[0]+timedelta(days=j) for j in range(-7, 7+1)]+[seq_of_dates[-1]+timedelta(days=j) for j in range(-7, 7+1)]


                if sorted_dates[j][1] == 'rt':
                    seq_of_dates = []
                        ## we will use number of doses of rt. 
                    if pd.isna(row['DOSBOOSTRT']) == False:
                        seq_of_dates = pd.date_range(start = sorted_dates[j][0][0], periods = row['DOSBOOSTRT'])
                    elif pd.isna(row['DOSGPRT']) == False:
                        seq_of_dates = pd.date_range(start = sorted_dates[j][0][0], periods = row['DOSGPRT'])
                    else:
                        # 1 cure per day for 4-6 weeks (5 is choosen)
                        if pd.isna(sorted_dates[j][0][1]) == True:
                            seq_of_dates = pd.date_range(start = sorted_dates[j][0][0], end= sorted_dates[j][0][0]+timedelta(weeks=5))
                        else:
                            seq_of_dates = pd.date_range(start = sorted_dates[j][0][0], end=sorted_dates[j][0][1])

                    seq_of_dates = [seq_of_dates[i]+timedelta(days=j) for j in range(-7, 7+1) for i in range(len(seq_of_dates))]


                if sorted_dates[j][1] == 'antiher2':
                    date_antiher2_index = 0
                    seq_of_dates = []

                    while date_antiher2_index <= len(sorted_dates[j][0])-1:
                        if pd.isna(sorted_dates[j][0][date_antiher2_index+1]) == False:
                            # if date_fin is available compute number of cures, num_days/21. Check if the cure is done each 3 weeks.

                            num_cures = (str_to_date(sorted_dates[j][0][date_antiher2_index+1]) - str_to_date(sorted_dates[j][0][date_antiher2_index])).days/21
                            num_cures = int(np.ceil(num_cures))
                            seq_of_dates = seq_of_dates + ([sorted_dates[j][0][date_antiher2_index]+timedelta(days=n*21) for n in range(num_cures)])

                        else:
                            if pd.isna(sorted_dates[j][0][date_antiher2_index]) == True:
                                pass

                            else:
                            # if no date_end is specified, take 18 injections
                            # 18 injections each 3 weeks (combined with CT) (verifie ca)
                                
                                seq_of_dates = [sorted_dates[j][0][date_antiher2_index]+timedelta(days=n*21) for n in range(18)]

                        date_antiher2_index+=2

                    seq_of_dates = [seq_of_dates[i]+timedelta(days=j) for j in range(-7, 7+1) for i in range(len(seq_of_dates))]

                if sorted_dates[j][1] == 'ct':
                        seq_of_dates = []
                        date_ct_index = 0

                        while date_ct_index <= len(sorted_dates[j][0]) - 1:
                            periods = 21 #CT each 21days
                            if pd.isna(sorted_dates[j][0][date_ct_index]) == True:
                                pass

                            else:
                              #  if pd.isna(sorted_dates[j][0][date_ct_index+1]) == False:
                                if date_ct_index == 0:
                                    num_cycles = np.max([row['NBCYCT.q1.f1'], row['NBCYCT.q2.f1'], row['NBCYCT.q3.f1'], row['NBCYCT.q4.f1'], row['NBCYCT.q5.f1']])

                                    if pd.isna(num_cycles):
                                        if 'TAXOL' in [row['NBCYCT.q1.f1'], row['NBCYCT.q2.f1'], row['NBCYCT.q3.f1'], row['NBCYCT.q4.f1'], row['NBCYCT.q5.f1']] :
                                            num_cycles = 11 
                                            periods = 7

                                        else: 
                                            num_cycles = 4

                                elif date_ct_index == 2:
                                    num_cycles = np.max([row['NBCYCT.q1.f2'], row['NBCYCT.q2.f2'],row['NBCYCT.q3.f2'], row['NBCYCT.q4.f2'],row['NBCYCT.q5.f2']])
                                    if pd.isna(num_cycles):
                                        if 'TAXOL' in [row['NBCYCT.q1.f1'], row['NBCYCT.q2.f1'], row['NBCYCT.q3.f1'], row['NBCYCT.q4.f1'], row['NBCYCT.q5.f1']] :

                                            num_cycles = 11 

                                        else: 
                                            num_cycles = 4

                                seq_of_dates =  seq_of_dates + [sorted_dates[j][0][date_ct_index]+timedelta(days=n*periods) for n in range(num_cycles)]
                                # knowing that CT last about 3 to 5 days we take 3days a window for each CT treatment 
                                seq_of_dates = [seq_of_dates[i]+timedelta(days=j) for j in range(-3, 3+1) for i in range(len(seq_of_dates))]

                            date_ct_index+=2


                if dict_therap['Date_event'] in seq_of_dates:
                   # print(index)
              #      data.loc[:,'therapies'].loc[index] += int_to_list(config[sorted_dates[j][1]])
                   # therapy = data.at[index, 'therapies']
                    #therapy += int_to_list(config[sorted_dates[j][1]])
                   
                    data.at[index, 'therapies'] += int_to_list(config[sorted_dates[j][1]])


    ## return list with no duplicates
    data.loc[:,'therapies'] = data['therapies'].apply(lambda x: list(dict.fromkeys(x)))

    ## remove 0 when there is therapies in the list 

    data.loc[:,'therapies'] = data['therapies'].apply(lambda x:  x[1:] if len(x)>1 else x)
    return data

def remove_duplicates(lst, element_to_remove):
    # Remove the specified element
    if len(lst)>1:
        lst = [item for item in lst if item != element_to_remove]
    # Remove duplicates
    unique_list = list(set(lst))
    return unique_list


def flatten_and_clean_list(lst):
    flattened = []
    for item in lst:
        if isinstance(item, list):
            flattened.extend(flatten_list(item))
        else:
            flattened.append(item)
            
    return remove_duplicates(flattened, 'pas de sous therapie')       


def cal_therapies(df, window=14):
                                   
    """ This function makes the therapies sequential using the date they've been administrated to display their code
    surgery : 1, 'ht':2, 'rt':3, 'ct':4, 'antiher2':5 
    data : the data with the therapies in binary form
    number_of_cycles: number of cycles of cures for treatments  # cf. 'nb_cycles_adj_ct'
    timestamps: The regularity of therapy session, each 5days: regular vs irregular
    ts: the number of days between two treatments or the sequence of days when it's irregular eq:[2, 5, 7]: 2, 5 and 7 days after the first treatment
    # Take into account the date_end of the treatment in the sequence ts 
    window : The number of days to add before and after the treatment in order to be matched with reports writen before and after the treatment"""
        
    data = df.copy()
    therapies = ['chirurgie', 'ht', 'rt', 'ct', 'antiher2']
    dates = [['dat_first_surgery', 'dat_second_surgery', 'dat_third_surgery'], ['DATDHT.q1','DATFHT.q1','DATDHT.q2','DATFHT.q2','DATDHT.q3','DATFHT.q3','DATDHT.q4','DATFHT.q4'], 
             ['DATDRT', 'DATFRT'], 
                ['DATDCT.f1', 'DATFCT.f1', 'DATDCT.f2', 'DATFCT.f2'], 
                ['dat_beg_first_antiher2', 'dat_end_first_antiher2', 'dat_beg_second_antiher2', 'dat_end_second_antiher2']]

    config = {'chirurgie':'chirurgie', 'ht':'hormone therapie', 'rt':'radiotherapie', 'ct':'chimiotherapie', 'antiher2':'adjuvant anti-her2', 'no therapy': 'pas de therapie'}
    
    data['therapies'] =  [['pas de therapie'] for r in range(len(data))]
    #data["therapies"] = [[] for r in range(len(data))]
    date_therapy = list()

    for index, row in data.iterrows(): 
        list_of_dates = []
        list_of_therapies = []

        dict_therap = dict()

        for therapy, date in zip(therapies, dates):
           
            if row[therapy] != 0:

                dict_therap['Num_dossier'] = row['Num_dossier']
                dict_therap[therapy]= row[therapy]

                #for d in date: 
              #  date_therapy.append(str_to_date(row[date]))
                dict_therap[tuple(date)] = [str_to_date(row[d]) for d in date]


                Date_exam = str_to_date(row['Date_exam'])
                dict_therap['Date_event'] = Date_exam


                list_of_therapies.append(therapy)
                list_of_dates.append([str_to_date(row[d]) for d in date])

        list_of_therapies.append('Date_event')
        list_of_dates.append([Date_exam])
        
        
        sorted_dates = sorted(zip(list_of_dates, list_of_therapies))
        
        if sorted_dates[0][1] == 'Date_event':

             ## If the less recent event between therapies and the current day, we don't have therapy yet = 0
        ## I need to see if within the BC window we will find the date_event (in a backward way)

            for j in range(len(sorted_dates)):

                if sorted_dates[j][1] != 'Date_event':
                    window_dates = list()
                   ## We assume that reports that happened 14 days before and 14 days after the date of the therapy (beg and end for certain therapies) are somehow concern therapies

                    for d in sorted_dates[j][0]:
                        window_dates = window_dates + [d+timedelta(days=i) for i in range(-window, window+1) if pd.isna(d)==False]

                    if dict_therap['Date_event'] in window_dates:
                        data.at[index, 'therapies'] += int_to_list(config[sorted_dates[j][1]])
                      #  print(int_to_list(config[sorted_dates[j][1]]))
                    

        for j in range(len(sorted_dates)):
            if sorted_dates[j][1] != 'Date_event':

                if sorted_dates[j][1] == 'chirurgie':
                    ## For each surgey, we use 14 days before and 14 days to display information about surgery on the sequence
                    seq_of_dates = []

                    for date_chir in sorted_dates[j][0]:
                        seq_of_dates = seq_of_dates + [date_chir+timedelta(days=i) for i in range(-window, window+1) if pd.isna(date_chir) == False]

                   # seq_of_dates = [seq_of_dates[i]+timedelta(days=j) for j in range(-window, window+1) for i in range(len(seq_of_dates))]


                if sorted_dates[j][1] == 'ht':
                    ## Normally for ht, we have 1 injection per day for 5, 7-10 years for patients with lymph nodes affected (from the therapeutic protocol). We will use it if the information number of doses is unavailable. 
                    seq_of_dates = []

                    date_ht_index = 0
                    while date_ht_index <= len(sorted_dates[j][0])-1:

                        if pd.isna(sorted_dates[j][0][date_ht_index+1]) == False:
                            
                            seq_of_dates = seq_of_dates + list(pd.date_range(start = sorted_dates[j][0][date_ht_index], end=sorted_dates[j][0][date_ht_index+1]))

                        else: 
                            if pd.isna(sorted_dates[j][0][date_ht_index]) == True:
                                pass

                            else:

                                if row['pnuicc_4cl'] == 1 or pd.isna(row['pnuicc_4cl']) == True:  # if 0 nodes involved or missing information (520patients)
                                    seq_of_dates = seq_of_dates + list(pd.date_range(start = sorted_dates[j][0][date_ht_index], end = sorted_dates[j][0][date_ht_index]+timedelta(days=5*365.2425)))
                                else: 
                                    seq_of_dates = seq_of_dates + list(pd.date_range(start = sorted_dates[j][0][date_ht_index], end= sorted_dates[j][0][date_ht_index]+timedelta(days=8.5*365.2425)))

                        date_ht_index+=2

                    seq_of_dates = seq_of_dates + [seq_of_dates[0]+timedelta(days=j) for j in range(-7, 7+1)]+[seq_of_dates[-1]+timedelta(days=j) for j in range(-7, 7+1)]


                if sorted_dates[j][1] == 'rt':
                    seq_of_dates = []
                        ## we will use number of doses of rt. 
                    if pd.isna(row['DOSBOOSTRT']) == False:
                        seq_of_dates = pd.date_range(start = sorted_dates[j][0][0], periods = row['DOSBOOSTRT'])
                    elif pd.isna(row['DOSGPRT']) == False:
                        seq_of_dates = pd.date_range(start = sorted_dates[j][0][0], periods = row['DOSGPRT'])
                    else:
                        # 1 cure per day for 4-6 weeks (5 is choosen)
                        if pd.isna(sorted_dates[j][0][1]) == True:
                            seq_of_dates = pd.date_range(start = sorted_dates[j][0][0], end= sorted_dates[j][0][0]+timedelta(weeks=5))
                        else:
                            seq_of_dates = pd.date_range(start = sorted_dates[j][0][0], end=sorted_dates[j][0][1])

                    seq_of_dates = [seq_of_dates[i]+timedelta(days=j) for j in range(-7, 7+1) for i in range(len(seq_of_dates))]


                if sorted_dates[j][1] == 'antiher2':
                    date_antiher2_index = 0
                    seq_of_dates = []

                    while date_antiher2_index <= len(sorted_dates[j][0])-1:
                        if pd.isna(sorted_dates[j][0][date_antiher2_index+1]) == False:
                            # if date_fin is available compute number of cures, num_days/21. Check if the cure is done each 3 weeks.

                            num_cures = (str_to_date(sorted_dates[j][0][date_antiher2_index+1]) - str_to_date(sorted_dates[j][0][date_antiher2_index])).days/21
                            num_cures = int(np.ceil(num_cures))
                            seq_of_dates = seq_of_dates + ([sorted_dates[j][0][date_antiher2_index]+timedelta(days=n*21) for n in range(num_cures)])

                        else:
                            if pd.isna(sorted_dates[j][0][date_antiher2_index]) == True:
                                pass

                            else:
                            # if no date_end is specified, take 18 injections
                            # 18 injections each 3 weeks (combined with CT) (verifie ca)
                              
                                seq_of_dates = [sorted_dates[j][0][date_antiher2_index]+timedelta(days=n*21) for n in range(18)]

                        date_antiher2_index+=2

                    seq_of_dates = [seq_of_dates[i]+timedelta(days=j) for j in range(-7, 7+1) for i in range(len(seq_of_dates))]

                if sorted_dates[j][1] == 'ct':
                        seq_of_dates = []
                        date_ct_index = 0

                        while date_ct_index <= len(sorted_dates[j][0]) - 1:
                            periods = 21 #CT each 21days
                            if pd.isna(sorted_dates[j][0][date_ct_index]) == True:
                                pass

                            else:
                              #  if pd.isna(sorted_dates[j][0][date_ct_index+1]) == False:
                            
                                if date_ct_index == 0:
                                    
                                    num_cycles = np.max([row['NBCYCT.q1.f1'], row['NBCYCT.q2.f1'], row['NBCYCT.q3.f1'], row['NBCYCT.q4.f1'], row['NBCYCT.q5.f1']])

                                    if pd.isna(num_cycles):
                                        if 'TAXOL' in [row['NBCYCT.q1.f1'], row['NBCYCT.q2.f1'], row['NBCYCT.q3.f1'], row['NBCYCT.q4.f1'], row['NBCYCT.q5.f1']] :
                                            num_cycles = 11 
                                            periods = 7

                                        else: 
                                            num_cycles = 4

                                elif date_ct_index == 2:
                                    num_cycles = np.max([row['NBCYCT.q1.f2'], row['NBCYCT.q2.f2'],row['NBCYCT.q3.f2'], row['NBCYCT.q4.f2'],row['NBCYCT.q5.f2']])
                                    if pd.isna(num_cycles):
                                        if 'TAXOL' in [row['NBCYCT.q1.f1'], row['NBCYCT.q2.f1'], row['NBCYCT.q3.f1'], row['NBCYCT.q4.f1'], row['NBCYCT.q5.f1']] :

                                            num_cycles = 11 

                                        else: 
                                            num_cycles = 4

                                seq_of_dates =  seq_of_dates + [sorted_dates[j][0][date_ct_index]+timedelta(days=n*periods) for n in range(int(num_cycles))]
                                # knowing that CT last about 3 to 5 days we take 3days a window for each CT treatment 
                                seq_of_dates = [seq_of_dates[i]+timedelta(days=j) for j in range(-3, 3+1) for i in range(len(seq_of_dates))]

                            date_ct_index+=2


                if dict_therap['Date_event'] in seq_of_dates:
                   # print(index)
              #      data.loc[:,'therapies'].loc[index] += int_to_list(config[sorted_dates[j][1]])
                   # therapy = data.at[index, 'therapies']
                    #therapy += int_to_list(config[sorted_dates[j][1]])
                   
                    data.at[index, 'therapies'] += int_to_list(config[sorted_dates[j][1]])


    ## return list with no duplicates
    data.loc[:,'therapies'] = data['therapies'].apply(lambda x: list(dict.fromkeys(x)))

    ## remove 0 when there is therapies in the list 

    data.loc[:,'therapies'] = data['therapies'].apply(lambda x:  x[1:] if len(x)>1 else x)
    return data
def cal_subtherapies(data, window=14):

    therapies = ['chirurgie', 'ht', 'rt', 'ct', 'antiher2']
    dates = [['dat_first_surgery', 'dat_second_surgery', 'dat_third_surgery'], ['DATDHT.q1','DATFHT.q1','DATDHT.q2','DATFHT.q2','DATDHT.q3','DATFHT.q3','DATDHT.q4','DATFHT.q4'], 
             ['DATDRT', 'DATFRT'], 
                ['DATDCT.f1', 'DATFCT.f1', 'DATDCT.f2', 'DATFCT.f2'], 
                ['dat_beg_first_antiher2', 'dat_end_first_antiher2', 'dat_beg_second_antiher2', 'dat_end_second_antiher2']]


    subtherapies = {'chirurgie' :['surg_first_act_completed', 'surg_second_act_completed', 'surg_third_act_completed'],
                   'ht': ['TYPHT.q1_v2','TYPHT.q2_v2','TYPHT.q3_v2','TYPHT.q4_v2'],
                   'rt': ['RT_subtype'], 'ct': ['typct_1e_instance', 'typct_2e_instance'],
                    'antiher2': ['antiher2_molecule_first_instance', 'antiher2_molecule_second_instance']}

    data['sub_therapies'] =  [['pas de sous therapie'] for r in range(len(data))]


    date_therapy = list()

    for index, row in data.iterrows(): 
        list_of_dates = []
        list_of_therapies = []

        dict_therap = dict()

        for therapy, date in zip(therapies, dates):
            if row[therapy] != 0:

                dict_therap['Num_dossier'] = row['Num_dossier']
                dict_therap[therapy]= row[therapy]

                #for d in date: 
              #  date_therapy.append(str_to_date(row[date]))
                dict_therap[tuple(date)] = [str_to_date(row[d]) for d in date]


                Date_exam = str_to_date(row['Date_exam'])
                dict_therap['Date_event'] = Date_exam


                list_of_therapies.append(therapy)
                list_of_dates.append([str_to_date(row[d]) for d in date])

        list_of_therapies.append('Date_event')
        list_of_dates.append([Date_exam])

        sorted_dates = sorted(zip(list_of_dates, list_of_therapies))
        if sorted_dates[0][1] == 'Date_event':
             ## If the less recent event between therapies and the current day, we don't have therapy yet = 0
        ## I need to see if within the BC window we will find the date_event (in a backward way)

            for j in range(len(sorted_dates)):

                if sorted_dates[j][1] != 'Date_event':
                    window_dates = list()
                   ## We assume that reports that happened 14 days before and 14 days after the date of the therapy (beg and end for certain therapies) are somehow concern therapies

                    for d in sorted_dates[j][0]:
                        window_dates.append([d+timedelta(days=i) for i in range(-window, window+1) if pd.isna(d)==False])

                    step = 0
                    while step < len(window_dates):
                        if dict_therap['Date_event'] in window_dates[step]:
                            data.at[index, 'sub_therapies'] += row[subtherapies[sorted_dates[j][1]][step]]
                        step +=1

        for j in range(len(sorted_dates)):
            if sorted_dates[j][1] != 'Date_event':

                if sorted_dates[j][1] == 'chirurgie':
                    ## For each surgey, we use 14 days before and 14 days to display information about surgery on the sequence

                    step=0
                    while step < len(sorted_dates[j][0]):

                        seq_of_dates = []
                        seq_of_dates = [sorted_dates[j][0][step]+timedelta(days=i) for i in range(-window, window+1) if pd.isna(sorted_dates[j][0][step]) == False]

                        if dict_therap['Date_event'] in seq_of_dates:

                          #  print(dict_therap['Date_event'])
                         #   print(seq_of_dates)
                         #   print(index)

                            data.at[index, 'sub_therapies'] += row[subtherapies[sorted_dates[j][1]][step]]

                        step+=1

                if sorted_dates[j][1] == 'ht':
                    ## Normally for ht, we have 1 injection per day for 5, 7-10 years for patients with lymph nodes affected (from the therapeutic protocol). We will use it if the information number of doses is unavailable. 
                    seq_of_dates = []
                    date_ht_index = 0
                    step = 0
                    while date_ht_index <= len(sorted_dates[j][0])-1:
                       # if pd.isna(sorted_dates[j][0][date_ht_index]) == False:

                        if pd.isna(sorted_dates[j][0][date_ht_index+1]) == False:
                            seq_of_dates = list(pd.date_range(start = sorted_dates[j][0][date_ht_index], end=sorted_dates[j][0][date_ht_index+1]))

                        else: 
                            if pd.isna(sorted_dates[j][0][date_ht_index]) == True:
                                pass

                            else:

                                if row['pnuicc_4cl'] == 1 or pd.isna(row['pnuicc_4cl']) == True:  # if 0 nodes involved or missing information (520patients)
                                    seq_of_dates = list(pd.date_range(start = sorted_dates[j][0][date_ht_index], end = sorted_dates[j][0][date_ht_index]+timedelta(days=5*365.2425)))
                                else: 
                                    seq_of_dates = list(pd.date_range(start = sorted_dates[j][0][date_ht_index], end= sorted_dates[j][0][date_ht_index]+timedelta(days=8.5*365.2425)))
                      
                            seq_of_dates = seq_of_dates + [seq_of_dates[0]+timedelta(days=j) for j in range(-7, 7+1)]+[seq_of_dates[-1]+timedelta(days=j) for j in range(-7, 7+1)]
                      
                           
                      #  print(seq_of_dates[-10:])
                      #  print(date_ht_index)
                    #    print()

                        if dict_therap['Date_event'] in seq_of_dates:
                        
                            if pd.isna(row[subtherapies[sorted_dates[j][1]][step]]): ## when I can have many molec at the same time make sure I don't add nan 
                                pass
                            else:
                                data.at[index, 'sub_therapies'] += row[subtherapies[sorted_dates[j][1]][step]]

                        date_ht_index+=2
                        step+=1

                if sorted_dates[j][1] == 'rt':
                    seq_of_dates = []
                        ## we will use number of doses of rt. 
                    if pd.isna(row['DOSBOOSTRT']) == False:
                        seq_of_dates = pd.date_range(start = sorted_dates[j][0][0], periods = row['DOSBOOSTRT'])
                    elif pd.isna(row['DOSGPRT']) == False:
                        seq_of_dates = pd.date_range(start = sorted_dates[j][0][0], periods = row['DOSGPRT'])
                    else:
                        # 1 cure per day for 4-6 weeks (5 is choosen)
                        if pd.isna(sorted_dates[j][0][1]) == True:
                            seq_of_dates = pd.date_range(start = sorted_dates[j][0][0], end= sorted_dates[j][0][0]+timedelta(weeks=5))
                        else:
                            seq_of_dates = pd.date_range(start = sorted_dates[j][0][0], end=sorted_dates[j][0][1])

                    seq_of_dates = [seq_of_dates[i]+timedelta(days=j) for j in range(-7, 7+1) for i in range(len(seq_of_dates))]

                    if dict_therap['Date_event'] in seq_of_dates:
                        data.at[index, 'sub_therapies'] += row[subtherapies[sorted_dates[j][1]]][0]


                if sorted_dates[j][1] == 'antiher2':
                    date_antiher2_index = 0
                    step = 0
                    seq_of_dates = []

                    while date_antiher2_index <= len(sorted_dates[j][0])-1:
                        if pd.isna(sorted_dates[j][0][date_antiher2_index+1]) == False:
                            # if date_fin is available compute number of cures, num_days/21. Check if the cure is done each 3 weeks.

                            num_cures = (str_to_date(sorted_dates[j][0][date_antiher2_index+1]) - str_to_date(sorted_dates[j][0][date_antiher2_index])).days/21
                            num_cures = int(np.ceil(num_cures))
                            seq_of_dates = ([sorted_dates[j][0][date_antiher2_index]+timedelta(days=n*21) for n in range(num_cures)])

                        else:
                            if pd.isna(sorted_dates[j][0][date_antiher2_index]) == True:
                                pass

                            else:
                            # if no date_end is specified, take 18 injections
                            # 18 injections each 3 weeks (combined with CT) (verifie ca)
                                seq_of_dates = [sorted_dates[j][0][date_antiher2_index]+timedelta(days=n*21) for n in range(18)]

                        seq_of_dates = [seq_of_dates[i]+timedelta(days=j) for j in range(-7, 7+1) for i in range(len(seq_of_dates))]

                        if dict_therap['Date_event'] in seq_of_dates:
                            if isinstance(row[subtherapies[sorted_dates[j][1]][step]], list):
                                if len(row[subtherapies[sorted_dates[j][1]][step]]) >= 1:
                                    data.at[index, 'sub_therapies'] += row[subtherapies[sorted_dates[j][1]][step]]
                               # else:
                                 #   data.at[index, 'sub_therapies'].append(row[subtherapies[sorted_dates[j][1]][step]])
                                    

                            elif pd.isna(row[subtherapies[sorted_dates[j][1]][step]]):
                                pass
                                #data.at[index, 'sub_therapies'] += row[subtherapies[sorted_dates[j][1]][step]]

                        date_antiher2_index+=2
                        step+=1



                if sorted_dates[j][1] == 'ct':
                        seq_of_dates = []
                        date_ct_index = 0
                        step = 0
                        while date_ct_index <= len(sorted_dates[j][0]) - 1:
                            periods = 21 #CT each 21days
                            if pd.isna(sorted_dates[j][0][date_ct_index]) == True:
                                pass

                            else:
                              #  if pd.isna(sorted_dates[j][0][date_ct_index+1]) == False:
                                if date_ct_index == 0:
                                    num_cycles = np.max([row['NBCYCT.q1.f1'], row['NBCYCT.q2.f1'], row['NBCYCT.q3.f1'], row['NBCYCT.q4.f1'], row['NBCYCT.q5.f1']])

                                    if pd.isna(num_cycles):
                                        if 'TAXOL' in [row['NBCYCT.q1.f1'], row['NBCYCT.q2.f1'], row['NBCYCT.q3.f1'], row['NBCYCT.q4.f1'], row['NBCYCT.q5.f1']] :
                                            num_cycles = 11 
                                            periods = 7

                                        else: 
                                            num_cycles = 4

                                elif date_ct_index == 2:
                                    num_cycles = np.max([row['NBCYCT.q1.f2'], row['NBCYCT.q2.f2'],row['NBCYCT.q3.f2'], row['NBCYCT.q4.f2'],row['NBCYCT.q5.f2']])
                                    if pd.isna(num_cycles):
                                        if 'TAXOL' in [row['NBCYCT.q1.f1'], row['NBCYCT.q2.f1'], row['NBCYCT.q3.f1'], row['NBCYCT.q4.f1'], row['NBCYCT.q5.f1']] :
                                            num_cycles = 11 
                                        else: 
                                            num_cycles = 4

                                seq_of_dates =  seq_of_dates + [sorted_dates[j][0][date_ct_index]+timedelta(days=n*periods) for n in range(int(num_cycles))]
                                # knowing that CT last about 3 to 5 days we take 3days a window for each CT treatment 
                                seq_of_dates = seq_of_dates = [seq_of_dates[i]+timedelta(days=j) for j in range(-3, 3+1) for i in range(len(seq_of_dates))]

                            if dict_therap['Date_event'] in seq_of_dates: 
                                data.at[index, 'sub_therapies'] += row[subtherapies[sorted_dates[j][1]][step]]

                            date_ct_index+=2
                            step+=1


    ## return list with no duplicates
    data.loc[:,'sub_therapies'] = data['sub_therapies'].apply(flatten_and_clean_list)
                                   
    return data



def cal_therapies(df, window=14):
                                   
    """ This function makes the therapies sequential using the date they've been administrated to display their code
    surgery : 1, 'ht':2, 'rt':3, 'ct':4, 'antiher2':5 
    data : the data with the therapies in binary form
    number_of_cycles: number of cycles of cures for treatments  # cf. 'nb_cycles_adj_ct'
    timestamps: The regularity of therapy session, each 5days: regular vs irregular
    ts: the number of days between two treatments or the sequence of days when it's irregular eq:[2, 5, 7]: 2, 5 and 7 days after the first treatment
    # Take into account the date_end of the treatment in the sequence ts 
    window : The number of days to add before and after the treatment in order to be matched with reports writen before and after the treatment"""
        
    data = df.copy()
    therapies = ['chirurgie', 'ht', 'rt', 'ct', 'antiher2']
    dates = [['dat_first_surgery', 'dat_second_surgery', 'dat_third_surgery'], ['DATDHT.q1','DATFHT.q1','DATDHT.q2','DATFHT.q2','DATDHT.q3','DATFHT.q3','DATDHT.q4','DATFHT.q4'], 
             ['DATDRT', 'DATFRT'], 
                ['DATDCT.f1', 'DATFCT.f1', 'DATDCT.f2', 'DATFCT.f2'], 
                ['dat_beg_first_antiher2', 'dat_end_first_antiher2', 'dat_beg_second_antiher2', 'dat_end_second_antiher2']]

    config = {'chirurgie':'chirurgie', 'ht':'hormone therapie', 'rt':'radiotherapie', 'ct':'chimiotherapie', 'antiher2':'adjuvant anti-her2', 'no therapy': 'pas de therapie'}
    
    data['therapies'] =  [['pas de therapie'] for r in range(len(data))]
    #data["therapies"] = [[] for r in range(len(data))]
    date_therapy = list()

    for index, row in data.iterrows(): 
        list_of_dates = []
        list_of_therapies = []

        dict_therap = dict()

        for therapy, date in zip(therapies, dates):
           
            if row[therapy] != 0:

                dict_therap['Num_dossier'] = row['Num_dossier']
                dict_therap[therapy]= row[therapy]

                #for d in date: 
              #  date_therapy.append(str_to_date(row[date]))
                dict_therap[tuple(date)] = [str_to_date(row[d]) for d in date]


                Date_exam = str_to_date(row['Date_exam'])
                dict_therap['Date_event'] = Date_exam


                list_of_therapies.append(therapy)
                list_of_dates.append([str_to_date(row[d]) for d in date])

        list_of_therapies.append('Date_event')
        list_of_dates.append([Date_exam])
        
        
        sorted_dates = sorted(zip(list_of_dates, list_of_therapies))
        
        if sorted_dates[0][1] == 'Date_event':

             ## If the less recent event between therapies and the current day, we don't have therapy yet = 0
        ## I need to see if within the BC window we will find the date_event (in a backward way)

            for j in range(len(sorted_dates)):

                if sorted_dates[j][1] != 'Date_event':
                    window_dates = list()
                   ## We assume that reports that happened 14 days before and 14 days after the date of the therapy (beg and end for certain therapies) are somehow concern therapies

                    for d in sorted_dates[j][0]:
                        window_dates = window_dates + [d+timedelta(days=i) for i in range(-window, window+1) if pd.isna(d)==False]

                    if dict_therap['Date_event'] in window_dates:
                        data.at[index, 'therapies'] += int_to_list(config[sorted_dates[j][1]])
                      #  print(int_to_list(config[sorted_dates[j][1]]))
                    

        for j in range(len(sorted_dates)):
            if sorted_dates[j][1] != 'Date_event':

                if sorted_dates[j][1] == 'chirurgie':
                    ## For each surgey, we use 14 days before and 14 days to display information about surgery on the sequence
                    seq_of_dates = []

                    for date_chir in sorted_dates[j][0]:
                        seq_of_dates = seq_of_dates + [date_chir+timedelta(days=i) for i in range(-window, window+1) if pd.isna(date_chir) == False]

                   # seq_of_dates = [seq_of_dates[i]+timedelta(days=j) for j in range(-window, window+1) for i in range(len(seq_of_dates))]


                if sorted_dates[j][1] == 'ht':
                    ## Normally for ht, we have 1 injection per day for 5, 7-10 years for patients with lymph nodes affected (from the therapeutic protocol). We will use it if the information number of doses is unavailable. 
                    seq_of_dates = []

                    date_ht_index = 0
                    while date_ht_index <= len(sorted_dates[j][0])-1:

                        if pd.isna(sorted_dates[j][0][date_ht_index+1]) == False:
                         #   print(row['Num_dossier'], sorted_dates[j][0])
                            seq_of_dates = seq_of_dates + list(pd.date_range(start = sorted_dates[j][0][date_ht_index], end=sorted_dates[j][0][date_ht_index+1]))
                            

                        else: 
                            if pd.isna(sorted_dates[j][0][date_ht_index]) == True:
                                pass

                            else:

                                if row['pnuicc_4cl'] == 1 or pd.isna(row['pnuicc_4cl']) == True:  # if 0 nodes involved or missing information (520patients)
                                    seq_of_dates = seq_of_dates + list(pd.date_range(start = sorted_dates[j][0][date_ht_index], end = sorted_dates[j][0][date_ht_index]+timedelta(days=5*365.2425)))
                                else: 
                                    seq_of_dates = seq_of_dates + list(pd.date_range(start = sorted_dates[j][0][date_ht_index], end= sorted_dates[j][0][date_ht_index]+timedelta(days=8.5*365.2425)))

                        date_ht_index+=2

                    seq_of_dates = seq_of_dates + [seq_of_dates[0]+timedelta(days=j) for j in range(-7, 7+1)]+[seq_of_dates[-1]+timedelta(days=j) for j in range(-7, 7+1)]


                if sorted_dates[j][1] == 'rt':
                    seq_of_dates = []
                        ## we will use number of doses of rt. 
                    if pd.isna(row['DOSBOOSTRT']) == False:
                        seq_of_dates = pd.date_range(start = sorted_dates[j][0][0], periods = row['DOSBOOSTRT'])
                    elif pd.isna(row['DOSGPRT']) == False:
                        seq_of_dates = pd.date_range(start = sorted_dates[j][0][0], periods = row['DOSGPRT'])
                    else:
                        # 1 cure per day for 4-6 weeks (5 is choosen)
                        if pd.isna(sorted_dates[j][0][1]) == True:
                            seq_of_dates = pd.date_range(start = sorted_dates[j][0][0], end= sorted_dates[j][0][0]+timedelta(weeks=5))
                        else:
                            seq_of_dates = pd.date_range(start = sorted_dates[j][0][0], end=sorted_dates[j][0][1])

                    seq_of_dates = [seq_of_dates[i]+timedelta(days=j) for j in range(-7, 7+1) for i in range(len(seq_of_dates))]


                if sorted_dates[j][1] == 'antiher2':
                    date_antiher2_index = 0
                    seq_of_dates = []

                    while date_antiher2_index <= len(sorted_dates[j][0])-1:
                        if pd.isna(sorted_dates[j][0][date_antiher2_index+1]) == False:
                            # if date_fin is available compute number of cures, num_days/21. Check if the cure is done each 3 weeks.

                            num_cures = (str_to_date(sorted_dates[j][0][date_antiher2_index+1]) - str_to_date(sorted_dates[j][0][date_antiher2_index])).days/21
                            num_cures = int(np.ceil(num_cures))
                            seq_of_dates = seq_of_dates + ([sorted_dates[j][0][date_antiher2_index]+timedelta(days=n*21) for n in range(num_cures)])

                        else:
                            if pd.isna(sorted_dates[j][0][date_antiher2_index]) == True:
                                pass

                            else:
                            # if no date_end is specified, take 18 injections
                            # 18 injections each 3 weeks (combined with CT) (verifie ca)
                              
                                seq_of_dates = [sorted_dates[j][0][date_antiher2_index]+timedelta(days=n*21) for n in range(18)]

                        date_antiher2_index+=2

                    seq_of_dates = [seq_of_dates[i]+timedelta(days=j) for j in range(-7, 7+1) for i in range(len(seq_of_dates))]

                if sorted_dates[j][1] == 'ct':
                        seq_of_dates = []
                        date_ct_index = 0

                        while date_ct_index <= len(sorted_dates[j][0]) - 1:
                            periods = 21 #CT each 21days
                            if pd.isna(sorted_dates[j][0][date_ct_index]) == True:
                                pass

                            else:
                              #  if pd.isna(sorted_dates[j][0][date_ct_index+1]) == False:
                            
                                if date_ct_index == 0:
                                    
                                    num_cycles = np.max([row['NBCYCT.q1.f1'], row['NBCYCT.q2.f1'], row['NBCYCT.q3.f1'], row['NBCYCT.q4.f1'], row['NBCYCT.q5.f1']])

                                    if pd.isna(num_cycles):
                                        if 'TAXOL' in [row['NBCYCT.q1.f1'], row['NBCYCT.q2.f1'], row['NBCYCT.q3.f1'], row['NBCYCT.q4.f1'], row['NBCYCT.q5.f1']] :
                                            num_cycles = 11 
                                            periods = 7

                                        else: 
                                            num_cycles = 4

                                elif date_ct_index == 2:
                                    num_cycles = np.max([row['NBCYCT.q1.f2'], row['NBCYCT.q2.f2'],row['NBCYCT.q3.f2'], row['NBCYCT.q4.f2'],row['NBCYCT.q5.f2']])
                                    if pd.isna(num_cycles):
                                        if 'TAXOL' in [row['NBCYCT.q1.f1'], row['NBCYCT.q2.f1'], row['NBCYCT.q3.f1'], row['NBCYCT.q4.f1'], row['NBCYCT.q5.f1']] :

                                            num_cycles = 11 

                                        else: 
                                            num_cycles = 4

                                seq_of_dates =  seq_of_dates + [sorted_dates[j][0][date_ct_index]+timedelta(days=n*periods) for n in range(int(num_cycles))]
                                # knowing that CT last about 3 to 5 days we take 3days a window for each CT treatment 
                                seq_of_dates = [seq_of_dates[i]+timedelta(days=j) for j in range(-3, 3+1) for i in range(len(seq_of_dates))]

                            date_ct_index+=2


                if dict_therap['Date_event'] in seq_of_dates:
                   
                    data.at[index, 'therapies'] += int_to_list(config[sorted_dates[j][1]])


    ## return list with no duplicates
    data.loc[:,'therapies'] = data['therapies'].apply(lambda x : flatten_and_clean_list(x, to_remove='pas de therapie'))
                                                      
    return data

def remove_duplicates(lst, element_to_remove):
    # Remove the specified element
    if len(lst)>1:
        lst = [item for item in lst if item != element_to_remove]
    # Remove duplicates
    unique_list = list(set(lst))
    return unique_list


def flatten_and_clean_list(lst, to_remove='pas de sous therapie'):
    flattened = []
    for item in lst:
        if isinstance(item, list):
            flattened.extend(flatten_list(item))
        else:
            flattened.append(item)
            
    return remove_duplicates(flattened, to_remove)   

