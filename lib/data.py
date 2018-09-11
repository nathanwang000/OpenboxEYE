import numpy as np
from sklearn.externals import joblib
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class Mimic2(Dataset):

    def __init__(self, mode='total', random_risk=False,
                 expert_feature_only=False, duplicate=0,
                 two_stage=False, threshold=None, noise=0.01,
                 seed=42):
        '''
        mode in [dead, survivor, total]: ways to impute missingness
        '''
        self.path = 'data/mimic/' + mode + '_mean_finalset.csv'
        self.data = pd.read_csv(self.path)
        self.x = self.data.ix[:,2:]
        
        if expert_feature_only:
            self.r = Variable(torch.FloatTensor(list(map(lambda name: 1 \
                                                         if 'worst' in name else 0,
                                                         self.x.columns))))
            
            self.x = self.x.ix[:,self.r.nonzero().view(-1)]

        self.y = self.data['In-hospital_death']

        xtrain, self.xte, ytrain, self.yte = train_test_split(self.x,
                                                              self.y,
                                                              test_size=0.25,
                                                              random_state=42,
                                                              stratify=self.y)

        self.xtrain,self.xval,self.ytrain,self.yval = train_test_split(xtrain,
                                                                       ytrain,
                                                                       test_size=0.25,
                                                                       random_state=42,
                                                                       stratify=ytrain)


        # standardize model
        scaler = StandardScaler()
        scaler.fit(self.xtrain)
        self.xtrain = scaler.transform(self.xtrain)
        self.xval = scaler.transform(self.xval)
        self.xte = scaler.transform(self.xte)  
        self.ytrain = self.ytrain.values
        self.yval = self.yval.values
        self.yte = self.yte.values        

        # risk factors to use
        self.r = Variable(torch.FloatTensor(list(map(lambda name: 1 \
                                                     if 'worst' in name else 0,
                                                     self.x.columns))))
        if random_risk:
            np.random.seed(seed)
            r = np.random.permutation(self.r.data.numpy())            
            self.r = Variable(torch.from_numpy(r))

        # handle duplicate experiment
        for i in range(duplicate):
            nxtrain = self.xtrain + noise * np.random.randn(*self.xtrain.shape)
            nxval = self.xval + noise * np.random.randn(*self.xval.shape)
            nxte = self.xte + noise * np.random.randn(*self.xte.shape)
            self.xtrain = np.hstack([self.xtrain, nxtrain])
            self.xval = np.hstack([self.xval, nxval])
            self.xte = np.hstack([self.xte, nxte])
            self.r = torch.cat([self.r, self.r])

        # use training data to delete variables for two stage approach
        risk_set = set(self.r.nonzero().view(-1).data.numpy())
        if two_stage:
            kept = set(self.r.nonzero().view(-1).data.numpy())
            corr = np.corrcoef(self.xtrain.T)
            d = self.r.numel()
            for i in range(d):
                if i not in kept: # unknown variable
                    # determine if we should keep it
                    keep = True
                    for j in range(len(corr[i])):
                        if j in kept and corr[i, j] >= threshold:
                            keep = False
                            break
                    if keep:
                        kept.add(i) # later unknown correlated with this will not keep

            # organize the data and risk
            kept = list(kept)
            self.r = Variable(torch.FloatTensor([(1 if i in risk_set else 0)\
                                                 for i in kept]))
            self.xtrain = self.xtrain[:, kept]
            self.xval = self.xval[:, kept]
            self.xte = self.xte[:, kept]

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.xtrain[idx], self.ytrain[idx]

