import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyrenn as prn


datosIn = pd.read_csv("pacientesreal.csv", sep=',')
datosOut = pd.read_csv("pacientesrealtarget.csv", sep=',')


P = np.array(datosIn)
Y = np.array(datosOut)

filename = "nnBP.csv"
net=prn.loadNN(filename)

y = prn.NNOut(P,net)
ytest = prn.NNOut(P,net)
ytest = np.array(ytest)
print(ytest)

