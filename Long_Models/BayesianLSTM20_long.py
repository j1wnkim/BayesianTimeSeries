#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np 
import torch 
import matplotlib.pyplot as plt 
import torch 


# In[2]:


print(torch.__version__)
data = pd.read_csv("/home/jik19004/FilesToRun/ASOS_10_CT_stations_tmpc_demand_2011_2023.csv").drop(columns=["Unnamed: 0"])
data.head()


# In[3]:


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
DateTimeCol = [datetime.strptime(date, "%m/%d/%Y %H:%M") for date in DateTimeCol]


# In[4]:


train_start = "1/1/2017 0:00"
train_end = "12/31/2018 23:00"

val_start = "1/1/2019 0:00"
val_end = "12/31/2019 23:00"

test_start = "1/01/2020 0:00"
test_end = "12/31/2021 0:00"


date_train_start = datetime.strptime(train_start, "%m/%d/%Y %H:%M")
date_train_end = datetime.strptime(train_end, "%m/%d/%Y %H:%M")

date_val_start = datetime.strptime(val_start, "%m/%d/%Y %H:%M")
date_val_end = datetime.strptime(val_end, "%m/%d/%Y %H:%M")

date_test_start = datetime.strptime(test_start, "%m/%d/%Y %H:%M")
date_test_end = datetime.strptime(test_end, "%m/%d/%Y %H:%M")

DateTimeCol_num = np.array(DateTimeCol)

print(np.where(DateTimeCol_num == date_train_start)[0][0])
print(np.where(DateTimeCol_num == date_train_end)[0][0])


print(np.where(DateTimeCol_num == date_val_start)[0][0])
print(np.where(DateTimeCol_num == date_val_end)[0][0])

print(np.where(DateTimeCol_num == date_test_start)[0][0])
print(np.where(DateTimeCol_num == date_test_end)[0][0])


# In[5]:


for i in range(len(DateTimeCol)):
    date = DateTimeCol[i]
    if int(date.year == 2017) and int(date.month) == 1 and int(date.day) == 1 and int(date.hour) == 0: # 
        print("index for Jan 1, 2017: ", i)
    if int(date.year) == 2018 and int(date.month) == 12 and int(date.day) == 31 and int(date.hour) == 22:
        print("index for Dec 31, 2021: ", i)
    if int(date.year) == 2019 and int(date.month) == 1 and int(date.day) == 1 and int(date.hour) == 0: # 
        print("index for Jan 1, 2019: ", i)
    if int(date.year) == 2019 and int(date.month) == 12 and int(date.day) == 31 and int(date.hour) == 22:
        print("index for Dec 31, 2019 ", i)
    if int(date.year) == 2020 and int(date.month) == 1 and int(date.day) == 1 and int(date.hour) == 0: # 
        print("index for Jan 1, 2019: ", i)
    if int(date.year) == 2021 and int(date.month) == 5 and int(date.day) == 31 and int(date.hour) == 22: 
        print("index for Dec 31, 2019 ", i)
        break


# In[6]:
def return_sequences(data, outputData, input_n_steps, output_n_steps):
    X = []
    Y = []
    length = len(data)
    print(length)
    for i in range(0,length, 1):
        input_indx = i + input_n_steps
        #output_indx = input_indx + output_n_steps
        output_indx = i + output_n_steps
        if (input_indx >= len(data)): # we need to have equally split sequences.
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


# In[7]:


from sklearn.preprocessing import StandardScaler 
def splitDataAndScale(data, output, train_start = None, train_end = None, val_start = None, val_end = None, test_start = None, test_end = None):
    #train_end +=1
    #val_end += 1
    #test_end += 1
    TrainingData = (data.iloc[train_start: train_end + 1, :].copy())
    print(len(TrainingData))
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


# In[8]:


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


# In[9]:


print(TrainingData.shape)
print(ValidationData.shape)
print(TestingData.shape)


# In[10]:


from torch.nn.utils.rnn import PackedSequence
from typing import *
import torch.nn as nn 


class VariationalDropout(nn.Module):
    """
    Applies the same dropout mask across the temporal dimension
    See https://arxiv.org/abs/1512.05287 for more details.
    Note that this is not applied to the recurrent activations in the LSTM like the above paper.
    Instead, it is applied to the inputs and outputs of the recurrent layer.
    """
    def __init__(self, dropout: float, batch_first: Optional[bool]=False):
        super().__init__()
        self.dropout = dropout
        self.batch_first = batch_first

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.dropout <= 0.:
            return x

        is_packed = isinstance(x, PackedSequence)
        if is_packed:
            x, batch_sizes = x
            max_batch_size = int(batch_sizes[0])
        else:
            batch_sizes = None
            max_batch_size = x.size(0)

        # Drop same mask across entire sequence
        if self.batch_first:
            m = x.new_empty(max_batch_size, 1, x.size(2), requires_grad=False).bernoulli_(1 - self.dropout)
        else:
            m = x.new_empty(1, max_batch_size, x.size(2), requires_grad=False).bernoulli_(1 - self.dropout)
        x = x.masked_fill(m == 0, 0) / (1 - self.dropout)

        if is_packed:
            return PackedSequence(x, batch_sizes)
        else:
            return x

class LSTM(nn.LSTM):
    def __init__(self, *args, dropouti: float=0.,
                 dropoutw: float=0., dropouto: float=0.,
                 batch_first=True, unit_forget_bias=True, **kwargs):
        super().__init__(*args, **kwargs, batch_first=batch_first)
        self.unit_forget_bias = unit_forget_bias
        self.dropoutw = dropoutw
        self.input_drop = VariationalDropout(dropouti,
                                             batch_first=batch_first)
        self.output_drop = VariationalDropout(dropouto,
                                              batch_first=batch_first)
        self._init_weights()

    def _init_weights(self):
        """
        Use orthogonal init for recurrent layers, xavier uniform for input layers
        Bias is 0 except for forget gate
        """
        for name, param in self.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "bias" in name and self.unit_forget_bias:
                nn.init.zeros_(param.data)
                param.data[self.hidden_size:2 * self.hidden_size] = 1

    def _drop_weights(self):
        for name, param in self.named_parameters():
            if "weight_hh" in name:
                getattr(self, name).data = \
                    torch.nn.functional.dropout(param.data, p=self.dropoutw,
                                                training=self.training).contiguous()

    def forward(self, input, hx=None):
        self._drop_weights()
        input = self.input_drop(input)
        seq, state = super().forward(input, hx=hx)
        return self.output_drop(seq), state

# In[11]:


class BayesianModel(torch.nn.Module):
    def __init__(self, input_size, params1, params2, params3, num_layers, output_size):    
    #params1 = [conv_kernel_size, stride, max_kernel_size, LSTM_hidden_size, LSTM_num_layers]
    #params2 = [dropout_LSTM, dropout_FFN]
    #params3 = [hidden_size1, hidden_size2, hidden_size3]
        super(BayesianModel, self).__init__()        
        self.conv1D = torch.nn.Conv1d(in_channels = 18, out_channels = 18, kernel_size = params1[0], stride = params1[1])
        self.max1D = torch.nn.MaxPool1d(kernel_size = params1[2], stride = params1[1])
        self.dropout1 = torch.nn.Dropout(params2[1])
        self.input_size = input_size - params1[0] - params1[2] + 2
        self.BiLSTM = LSTM(input_size = self.input_size, hidden_size = params1[3], dropout = params2[0], bidirectional = False, num_layers = params1[4])
        
        layers = []
        input_size = params1[3] #params1[3] *2 
        num_units = 0 
        for i in range(num_layers):
            num_units = params3[i] 
            layers.append(torch.nn.Linear(input_size, num_units, bias = True))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(params2[1])) #also add in the dropout. 
            input_size = num_units
        
        self.intermediate_layers = torch.nn.Sequential(*layers)
        self.finalNN = torch.nn.Linear(num_units, output_size, bias = True)
        
    def forward(self, x):
        x = self.conv1D(x)
        x = self.max1D(x) 
        x = self.dropout1(x) 
        x = self.BiLSTM(x) 
        
        x = x[0]
        x = x[:, -1,:]
        #x = x[1][0]
        
        x = self.intermediate_layers(x) 
        x = self.finalNN(x) 
        return x 





# In[12]:


import optuna
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

# use the past 72 hours in advance and then predict the 1st hour, 6th hour, 12 hours!

def evaluate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for data, target in val_loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            loss = criterion(output, target)
            running_loss += loss.item() * target.size(0)
    return running_loss / len(val_loader.dataset)

def Train_and_Evaluate(train_loader, val_loader, device, params1, params2, params3, numEpochs, early_stop_epochs):
    model = BayesianModel(input_size = 12, params1 = params1, params2 = params2, params3 = params3, num_layers = 3, output_size = 1)
    model = model.to(device);
    LossFunction = torch.nn.L1Loss();
    best_val_loss = float('inf')
    Training_Loss = float('inf')
    early_stop_count = 0


    Optimizer = torch.optim.Adam(params = model.parameters())
    for epoch in range(0,numEpochs):
        model.train()
        Training_Loss = 0;
        total_samples = 0;
        for input, output in train_loader:
            input = input.to(device);
            #output = torch.squeeze(output, 1);
            output = output.to(device);
            predictedVal = model(input)
            predictedVal = torch.squeeze(predictedVal, 1)
            Optimizer.zero_grad();
            batchLoss = LossFunction(predictedVal, output);
            batchLoss.backward();
            Optimizer.step();
            Training_Loss += batchLoss * output.size(0) #* output.size(0);
            total_samples += output.size(0)
        Training_Loss = Training_Loss.item()/total_samples


        Validation_Loss = 0;
        print("passed ", epoch, "epoch", "Training Loss: ", Training_Loss," ", end = "")
        with torch.no_grad(): 
            model.train() 
            total_val_samples = 0 
            Validation_Loss = 0 
            for val_input, val_output in val_loader:
                val_input = val_input.to(device)
                #val_output = torch.squeeze(val_output,1);
                val_output = val_output.to(device)
                predictedVal = model(val_input)
                predictedVal = torch.squeeze(predictedVal, 1)
                Validation_Loss += LossFunction(val_output, predictedVal) * val_output.size(0)
                total_val_samples += val_output.size(0)
            Validation_Loss = Validation_Loss/total_val_samples
            print("Validation Loss: ", Validation_Loss)

            if Validation_Loss < best_val_loss:
                best_val_loss = Validation_Loss
                torch.save(model, "/home/jik19004/FilesToRun/BayesianBiDirectional/BayesianBiLSTMLong")
                early_stop_count = 0;   
            else:
                early_stop_count += 1
            if early_stop_count >= early_stop_epochs:
                return (Training_Loss, best_val_loss)
    return (Training_Loss, best_val_loss)

def evaluate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for data, target in val_loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            loss = criterion(output, target)
            running_loss += loss.item() * target.size(0)
    return running_loss / len(val_loader.dataset)


def evaluate2(model, val_loader, criterion, criterion2, device):
    num_experiments = 100
    criterion2 = criterion2.to(device)
    model.train()

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
            pred = model(val_input)
            pred = torch.squeeze(pred, 1)
            pred = torch.unsqueeze(pred, 0)
               
            for i in range(num_experiments - 1):
                predictedVal2 = model(val_input)
                predictedVal2 = torch.squeeze(predictedVal2, 1)
                predictedVal2 = torch.unsqueeze(predictedVal2, 0)
                pred = torch.cat([pred, predictedVal2], dim = 0)
            
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


# In[13]:


def Train_and_Evaluate2(train_loader, val_loader, device, params1, params2, numEpochs, early_stop_epochs):
    #num_layers, input_dim, hidden_unit1, hidden_unit2, output_unit, lastNeurons, batch_size, params, device = None
    model = BayesianModel(params1 = params1, params2 = params2, device = device)
    model = model.to(device);
    TrainEpochLoss = [] 
    ValidationEpochLoss = [] 
    LossFunction = torch.nn.L1Loss();
    best_val_loss = float('inf')
    early_stop_count = 0


    Optimizer = torch.optim.Adam(params = model.parameters())
    for epoch in range(0,numEpochs):
        model.train()
        Training_Loss = 0;
        total_samples = 0;
        for input, output in train_loader:
            input = input.to(device);
            output = torch.squeeze(output, 1);
            output = output.to(device);
            predictedVal = model(input)
            predictedVal = torch.squeeze(predictedVal, 1)
            Optimizer.zero_grad();
            batchLoss = LossFunction(predictedVal, output);
            batchLoss.backward();
            Optimizer.step();
            Training_Loss += batchLoss * output.size(0) #* output.size(0);
            total_samples += output.size(0)
        Training_Loss = Training_Loss.item()/total_samples
        TrainEpochLoss.append(Training_Loss)


        Validation_Loss = 0;
        print("passed ", epoch, "epoch", "Training Loss: ", Training_Loss," ", end = "")
        with torch.no_grad():
            model.eval()
            total_val_samples = 0;
            Validation_Loss = 0;
            for val_input, val_output in val_loader:
                val_input = val_input.to(device);
                val_output = torch.squeeze(val_output,1);
                val_output = val_output.to(device);
                predictedVal = model(val_input)
                predictedVal = torch.squeeze(predictedVal, 1)
                Validation_Loss += LossFunction(val_output, predictedVal) * val_output.size(0)
                total_val_samples += val_output.size(0)
            Validation_Loss = Validation_Loss.item()/total_val_samples
            print("Validation Loss: ", Validation_Loss)
            ValidationEpochLoss.append(Validation_Loss)

            if Validation_Loss < best_val_loss:
                best_val_loss = Validation_Loss
                torch.save(model, "/home/jik19004/FilesToRun/BayesianBiDirectional/BayesianBiLSTMLong")
                early_stop_count = 0;   
            else:
                early_stop_count +=1
            if early_stop_count >= early_stop_epochs:
                return (TrainEpochLoss, ValidationEpochLoss);
    return (TrainEpochLoss, ValidationEpochLoss);


# In[14]:


TrainingDataset = TimeSeriesDataset(np.array(TrainingData),np.array(TrainingOutput));
TrainingLoader = DataLoader(TrainingDataset, batch_size = 256, shuffle = True);


ValidationDataset = TimeSeriesDataset(ValidationData, ValidationOutput); ### Set it with the previous validation data
ValidationLoader = DataLoader(ValidationDataset, batch_size = 256, shuffle = False);


TestingDataset = TimeSeriesDataset(TestingData,TestingOutput); ### Set it with the previous testing data.
TestingLoader = DataLoader(TestingDataset, batch_size = 256, shuffle = False);


# In[15]:


params1 = [3, 1, 3, 64, 1]
params2 = [0.15, 0.15]
params3 = [64, 32, 16]
numEpochs = 3000 
early_stop_epochs = 100
#Train_and_Evaluate(TrainingLoader,ValidationLoader, torch.device("cuda"), params1, params2, params3, numEpochs, early_stop_epochs)


# In[16]:


from torchmetrics.regression import MeanAbsolutePercentageError

def LongTermEvaluate(model, ValidationData, ValidationOutput, DateTimeCol = DateTimeCol, index_start = 70117, index_end = 91291):
    Validation_Loss_MAE = []
    Validation_Loss_MAPE = []
    df_val = pd.DataFrame() 
    df_val_std = pd.DataFrame()
    
    for i in range(index_start, index_end, 24):       
        val_start = i
        val_end = val_start + 40 #val_start + 28
        val_output_start = DateTimeCol[val_start + 18]
        val_output_end = DateTimeCol[val_end + 1]

        if val_end < index_end:            
            ValidationDataset = TimeSeriesDataset(np.array(ValidationData[val_start - index_start: val_start - index_start + 24]), 
                                               np.array(ValidationOutput[val_start - index_start: val_start - index_start + 24]))
            ValidationLoader = DataLoader(ValidationDataset, batch_size = 3, shuffle = False) 
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


            val_str = val_output_start + "-" + val_output_end           
            val_loss = evaluate2(model, ValidationLoader, torch.nn.L1Loss(),MeanAbsolutePercentageError(), device)       
            print(val_loss[1].item())     
            Validation_Loss_MAE.append(val_loss[0].item())
            Validation_Loss_MAPE.append(val_loss[1].item())
            df_val = pd.concat([df_val, pd.DataFrame({val_str: val_loss[2]})], ignore_index=False, axis=1)
            df_val_std = pd.concat([df_val_std, pd.DataFrame({val_str: val_loss[3]})], ignore_index = False, axis =1 )
        else:
            return Validation_Loss_MAE, Validation_Loss_MAPE, df_val, df_val_std 
         
model = torch.load("/home/jik19004/FilesToRun/BayesianBiDirectional/BayesianBiLSTMLong")
WeatherData = pd.read_csv('/home/jik19004/FilesToRun/ASOS_10_CT_stations_tmpc_demand_2011_2023.csv').drop(columns = ["Unnamed: 0"])
DateTimeCol = WeatherData["Datetime"]
tuples = evaluate2(model, TestingLoader,torch.nn.L1Loss(), MeanAbsolutePercentageError(), device = torch.device("cuda"))
print(tuples[1])

tuples = LongTermEvaluate(model, TrainingData, TrainingOutput, DateTimeCol = DateTimeCol, index_start = 52603, index_end = 70121)
ValidationLoss_series = pd.DataFrame({"Training_MAE": tuples[0], "Training_MAPE": tuples[1]})
ValidationLoss_series.to_csv("/home/jik19004/FilesToRun/BayesianBiDirectional/TrainingLong/TrainingLossesLong.csv", index = False)
tuples[2].to_csv("/home/jik19004/FilesToRun/BayesianBiDirectional/TrainingLong/TrainingPredictionsLong.csv", index = False)
tuples[3].to_csv("/home/jik19004/FilesToRun/BayesianBiDirectional/TrainingLong/TrainingStdLong.csv", index = False)

tuples = LongTermEvaluate(model, ValidationData, ValidationOutput, DateTimeCol = DateTimeCol, index_start = 70123, index_end = 78881)
ValidationLoss_series = pd.DataFrame({"Validation_MAE": tuples[0], "Validation_MAPE": tuples[1]})
ValidationLoss_series.to_csv("/home/jik19004/FilesToRun/BayesianBiDirectional/ValidationLong/ValidationLossesLong.csv", index = False)
tuples[2].to_csv("/home/jik19004/FilesToRun/BayesianBiDirectional/ValidationLong/ValidationPredictionsLong.csv", index = False)
tuples[3].to_csv("/home/jik19004/FilesToRun/BayesianBiDirectional/ValidationLong/ValidationStdLong.csv", index = False)



tuples = LongTermEvaluate(model, TestingData, TestingOutput, DateTimeCol = DateTimeCol, index_start = 78883, index_end = 91289)
ValidationLoss_series = pd.DataFrame({"Testing_MAE": tuples[0], "Testing_MAPE": tuples[1]})
ValidationLoss_series.to_csv("/home/jik19004/FilesToRun/BayesianBiDirectional/TestingLong/TestingLossesLong.csv", index = False)
tuples[2].to_csv("/home/jik19004/FilesToRun/BayesianBiDirectional/TestingLong/TestingPredictionsLong.csv", index = False)
tuples[3].to_csv("/home/jik19004/FilesToRun/BayesianBiDirectional/TestingLong/TestingStdLong.csv", index = False)


test_losses = evaluate2(model, TestingLoader, torch.nn.L1Loss(), MeanAbsolutePercentageError(), torch.device("cuda"))
print(test_losses[1])


# In[ ]:




# In[ ]:


#params1 = [conv_kernel_size, stride, max_kernel_size, LSTM_hidden_size, LSTM_num_layers]
#params2 = [dropout_LSTM, dropout_FFN]
#params3 = [hidden_size1, hidden_size2, hidden_size3]

