import pandas as pd
import logging
import os

def prRed(skk):
    print("\033[91m{}\033[00m" .format(skk)) 
    
def prGreen(skk):
    print("\033[92m{}\033[00m" .format(skk))  

def switchonlog(path='./save/', logname:str=''):
    r'''set log configuration'''
    filepath = path+'{}.log'.format(logname)
    if (os.path.exists(filepath)):
        os.remove(filepath)
    logging.basicConfig(filename=filepath, level=logging.INFO)

def loginfo(info:str='', color:str='white'):
    r'''print and log'''
    if color == 'red':
        prRed(info)
    elif color == 'green':
        prGreen(info)
    else:
        print(info)
    logging.info(info)

def record_tocsv(name, path='./save/', **kwargs):
    epoch = [i for i in range(1, len(kwargs[list(kwargs.keys())[0]])+1)]
    df = pd.DataFrame(kwargs)     
    file_name = path + name + ".csv"    
    df.to_csv(file_name, index = False) 

def read_fromcsv(name, path='./save/'):
    df = pd.read_csv("{}{}.csv".format(path, name))
    return df

def record_toexcel(name, **kwargs):
    path = './save/'
    epoch = [i for i in range(1, len(kwargs[list(kwargs.keys())[0]])+1)]
    df = pd.DataFrame(kwargs)     
    file_name = path + name + ".xls"    
    df.to_excel(file_name, sheet_name= "Sheet1", index = False) 

def read_fromexcel(name):
    path = './save/'
    df = pd.read_excel("{}{}.xls".format(path, name), sheet_name="Sheet1") # Sheet1
    return df

def exceltocsv():
    path = './save/'
    name='client alex cifar layer 1'
    df = pd.read_excel("{}{}.xls".format(path, name)) # Sheet1
    df.to_csv("{}{}.csv".format(path, name), index = False)

if __name__ == '__main__':
    exceltocsv()
    


