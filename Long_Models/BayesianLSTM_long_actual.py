#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np 
import torch 
import matplotlib.pyplot as plt 
from torchmetrics.regression import MeanAbsolutePercentageError


# In[2]:


import torch 

print(torch.__version__)


# In[3]:


data = pd.read_csv("/home/jik19004/FilesToRun/ASOS_10_CT_stations_tmpc_demand_2011_2023.csv").drop(columns=["Unnamed: 0"])
data.head()


# In[4]:


from datetime import datetime 

WeatherData = pd.read_csv('/home/jik19004/FilesToRun/ASOS_10_CT_stations_tmpc_demand_2011_2023.csv').drop(columns = ["Unnamed: 0"])
WeatherData.ffill(inplace = True)
WeatherData.bfill(inplace = True) # fill in missing values with the previous value.
DateTimeCol = WeatherData["Datetime"]
HourCol = []
WeekDayorWeekEndCol = [] 

for date in DateTimeCol:
    date = datetime.strptime(date, "%m/%d/%Y %H:%M")
    HourCol.append(date.hour)
    if date.weekday() < 5:
        WeekDayorWeekEndCol.append(0)
    else:
        WeekDayorWeekEndCol.append(1)

WeatherData.drop(columns = ["Datetime"], inplace = True) # drop the datetime column. 


WeatherData.insert(0, "Hour", HourCol)
WeatherData.insert(1, "Weekday or Weekend", WeekDayorWeekEndCol)


# In[5]:


DateTimeCol = [datetime.strptime(date, "%m/%d/%Y %H:%M") for date in DateTimeCol]
for i in range(len(DateTimeCol)):
    date = DateTimeCol[i]
    if int(date.year == 2018) and int(date.month) == 12 and int(date.day) == 31 and int(date.hour) == 18: # start of short term data
        print("index for Dec 31, 2018: ", i)
    if int(date.year) == 2021 and int(date.month) == 12 and int(date.day) == 31 and int(date.hour) == 23: # end of short term data  
        print("index for Dec 31, 2021: ", i)
    if int(date.year) == 2021 and int(date.month) == 1 and int(date.day) == 1 and int(date.hour) == 0: # start of validation
        print("index for Jan 1, 2021: ", i)
    if int(date.year) == 2022 and int(date.month) == 12 and int(date.day) == 31 and int(date.hour) == 23: # end of validation 
        print("index for Dec 31, 2022 ", i)
        break


from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class TimeSeriesDataset(Dataset):
    def __init__(self, data, output):
        data = torch.tensor(data).float();
        output = torch.tensor(output).float()
        self.data = data
        self.output = output;

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx];
        y = self.output[idx];

        return x, y;

def evaluate2(val_loader, criterion, criterion2, device):
    num_experiments = 100 
    criterion2 = criterion2.to(device)
    with torch.no_grad():
        total_val_samples = 0;
        Validation_Loss_MAE = 0;
        Validation_Loss_MAPE = 0;
        predictions = []  
        std_list = [] 
        for val_input, val_output in val_loader:
            val_input = val_input.to(device);
            val_output = val_output.to(device);
            #Avgpred
                
            Avgpred = torch.mean(pred, dim = 0)
            stdev = torch.std(pred, dim = 0)
            predCsv = Avgpred.cpu().numpy()
            stdev = stdev.cpu().numpy()
            predictions.extend(predCsv) 
            std_list.extend(stdev)
            
            Validation_Loss_MAE += criterion(val_output, Avgpred) * val_output.size(0)
            Validation_Loss_MAPE += criterion2(val_output, Avgpred) * val_output.size(0)
            total_val_samples += val_output.size(0)
        Validation_Loss_MAE = Validation_Loss_MAE/total_val_samples
        Validation_Loss_MAPE = Validation_Loss_MAPE/total_val_samples 
        return Validation_Loss_MAE, Validation_Loss_MAPE, predictions, std_list   




def return_sequences(data, outputData, input_n_steps, output_n_steps):
    X = []
    Y = []
    length = len(data)
    for i in range(0,length, 1):
        input_indx = i + input_n_steps
        #output_indx = input_indx + output_n_steps
        output_indx = i + output_n_steps
        if (input_indx > len(data)): # we need to have equally split sequences. >=
            break               # The remaining data that cannot fit into a fixed
                                # sequence will immediately be cut!
        else:
            Xsample = data.iloc[i:input_indx, :] # get the previous data
            #Ysample = outputData[input_indx:output_indx]
            Ysample = outputData[i]
            X.append(Xsample)
            Y.append(Ysample)
    X = np.asarray(X).astype('float64')
    Y = np.asarray(Y).astype('float64')
    return (X, Y)


# In[10]:


def splitDataAndScale(data, output, train_start = None, train_end = None, val_start = None, val_end = None, test_start = None, test_end = None):
    #train_end +=1
    #val_end += 1
    #test_end += 1
    TrainingData = (data.iloc[train_start: train_end + 1, :].copy())
    TrainingCategories = TrainingData.iloc[:, [0,1]]
    TrainingNumerical = TrainingData.iloc[:, 2:]
    TrainingOutput = output[train_start + 18: train_end + 2].copy()  
    Scaler = StandardScaler().fit(TrainingNumerical)
    TrainingNumerical = Scaler.transform(TrainingNumerical)
    TrainingCategories = TrainingCategories.reset_index(drop = True)
    TrainingData = pd.concat([TrainingCategories, pd.DataFrame(TrainingNumerical)], axis = 1)
    TrainingData.reset_index(drop = True, inplace = True)
    TrainingOutput.reset_index(drop = True, inplace = True)
    
    ValidationData = data.iloc[val_start: val_end + 1, :].copy()
    ValidationData.reset_index(drop = True, inplace = True)
    ValidationCategories = ValidationData.iloc[:, [0,1]]
    ValidationNumerical = ValidationData.iloc[:, 2:]
    ValidationNumerical = Scaler.transform(ValidationNumerical)
    ValidationCategories = ValidationCategories.reset_index(drop = True)
    ValidationData = pd.concat([ValidationCategories, pd.DataFrame(ValidationNumerical)], axis = 1)
    ValidationOutput = output[val_start + 18: val_end + 2].copy()
    ValidationData.reset_index(drop = True, inplace = True)
    ValidationOutput.reset_index(drop = True, inplace = True)
    
    TestingData = data.iloc[test_start: test_end + 1, :].copy()
    TestingData.reset_index(drop = True, inplace = True)
    TestingCategories = TestingData.iloc[:, [0,1]]
    TestingNumerical = TestingData.iloc[:, 2:]
    TestingNumerical = Scaler.transform(TestingNumerical)
    TestingCategories = TestingCategories.reset_index(drop = True)
    TestingData = pd.concat([TestingCategories, pd.DataFrame(TestingNumerical)], axis = 1)
    TestingOutput = output[test_start + 18: test_end + 2].copy()
    TestingData.reset_index(drop = True, inplace = True)
    TestingOutput.reset_index(drop = True, inplace = True)


    TrainingSequences = return_sequences(TrainingData, TrainingOutput, 18, 1)

    TransformedTrainingData = TrainingSequences[0]
    TransformedTrainingOutput = TrainingSequences[1]

    ValidationSequences = return_sequences(ValidationData, ValidationOutput, 18, 1)

    TransformedValidationData = ValidationSequences[0]
    TransformedValidationOutput = ValidationSequences[1]

    TestingSequences = return_sequences(TestingData, TestingOutput, 18, 1)

    TransformedTestingData = TestingSequences[0]
    TransformedTestingOutput = TestingSequences[1]


    return (TransformedTrainingData, TransformedTrainingOutput, TransformedValidationData, TransformedValidationOutput,
    TransformedTestingData, TransformedTestingOutput)


# In[11]:


from sklearn.preprocessing import StandardScaler 

DemandData = WeatherData['Demand'].copy() # The output data
WeatherData.drop(columns = ['Demand'], inplace = True)
data = splitDataAndScale(WeatherData, DemandData, train_start = 52603, train_end = 70121, val_start = 70123, val_end = 78881, test_start = 78883, test_end = 91289) # splitting the data into training, validation, and testing.


TrainingData = data[0]
TrainingOutput = data[1]

ValidationData = data[2]
ValidationOutput = data[3]

TestingData = data[4]
TestingOutput = data[5]

# In[12]:


from torch.nn.utils.rnn import PackedSequence
from typing import *
import torch.nn as nn 


# In[22]:



# In[31]:


Training_Loss_MAE = [] 
Validation_Loss_MAE = [] 
Testing_Loss_MAE = [] 

Training_Loss_MAPE = [] 
Validation_Loss_MAPE = [] 
Testing_Loss_MAPE = [] 
# 87668
df_train = pd.DataFrame()
df_val = pd.DataFrame() 
df_test = pd.DataFrame()

df_train_std = pd.DataFrame()
df_val_std = pd.DataFrame()
df_test_std = pd.DataFrame()

DateTimeCol = pd.read_csv("/home/jik19004/FilesToRun/ASOS_10_CT_stations_tmpc_demand_2011_2023.csv")["Datetime"]
ActualOutput = pd.read_csv("/home/jik19004/FilesToRun/ASOS_10_CT_stations_tmpc_demand_2011_2023.csv")["Demand"] 
#70338
#91291
def ParseActuals(DateTimeCol, ValidationOutput, start_index, end_index): 
    df_val = pd.DataFrame()
    for i in range(start_index, end_index, 24):
        val_start = i 
        val_end = val_start + 40
        val_output_start = DateTimeCol[val_start + 18]
        val_output_end = DateTimeCol[val_end + 1]
        
        if val_end < end_index:
            val_str = val_output_start + "-" + val_output_end 
            val_output = ValidationOutput[val_start - start_index: val_start - start_index + 24]
            df_val = pd.concat([df_val, pd.DataFrame({val_str: val_output})], ignore_index=False, axis=1)
        else:
            return df_val 


df_train = ParseActuals(DateTimeCol, TrainingOutput, start_index = 52603, end_index = 70121)
df_val = ParseActuals(DateTimeCol, ValidationOutput, start_index = 70123, end_index = 78881)
df_test = ParseActuals(DateTimeCol, TestingOutput, start_index = 78883, end_index = 91289)


df_train.to_csv("/home/jik19004/FilesToRun/BayesianBiDirectional/TrainingLong/TrainingActual_long.csv", index = False)
df_val.to_csv("/home/jik19004/FilesToRun/BayesianBiDirectional/ValidationLong/ValidationActual_long.csv", index = False)
df_test.to_csv("/home/jik19004/FilesToRun/BayesianBiDirectional/TestingLong/TestingActual_long.csv", index = False)


