import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyrenn as prn


datosIn = pd.read_csv("pacientestrain.csv", sep=',')
datosOut = pd.read_csv("pacientestarg.csv", sep=',')


P = np.array(datosIn)
Y = np.array(datosOut)

inputs = 7
outputs = 1

net = prn.CreateNN([inputs,10,14,outputs],dIn=[0],dIntern=[100],dOut=[10,20])
net = prn.train_LM(P,Y,net,verbose=True,k_max=20000,E_stop=1e-10)
y = prn.NNOut(P,net)
ytest = prn.NNOut(P,net)
ytest = np.array(ytest)
print(ytest)

filename = "nnBP.csv"
prn.saveNN(net, filename)



