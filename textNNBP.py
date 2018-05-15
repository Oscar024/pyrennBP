import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyrenn as prn


datosIn = pd.read_csv("pacientestrain.csv", sep=',')
datosOut = pd.read_csv("pacientestarg.csv", sep=',')


P = np.array(datosIn)
Y = np.array(datosOut)






filename = "nnBP.csv"
net=prn.loadNN(filename)

y = prn.NNOut(P,net)

IW, LW, b = prn.w2Wb(net)  # input-weight matrices,connection weight matrices, bias vectors

########################
# 1. Calculate NN Output
#Y_NN, n, a = prn.NNOut_(P, net, IW, LW, b, a=1, q0=0)
Y_NN = prn.NNOut(P,net)
########################
# 2. Calculate Cost function E
Y_delta = Y - Y_NN  # error matrix
e = np.reshape(Y_delta, (1, np.size(Y_delta)), order='F')[0]  # error vector
E = np.dot(e, e.transpose())  # Cost function (mean squared error)



print(E)

