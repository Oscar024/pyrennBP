import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyrenn as prn
import math


datosIn = pd.read_csv("pacientes200.csv", sep=',')
datosOut = pd.read_csv("pacientes200targ.csv", sep=',')


#P = np.array(datosIn)
#Y = np.array(datosOut)
#print(P.shape)
#print(Y.shape)


filename = "nnBP.csv"
net=prn.loadNN(filename)

edad = 83
sexo = 1
bmi = 49.95438563
sys = 107
dia = 78
fuma = 1
padres =0



target = 1-math.exp(-math.exp((math.log(4) - (22.949536 + (-0.156412*edad )+( -0.202933*sexo) + (-0.033881*bmi) + (-0.05933*sys) + (-0.128468*dia) + (-0.190731*fuma) +  (-0.166121*padres) + (0.001624*edad*dia))/0.876925)))

risk = target*100
print("Target: ")
print(risk)


P = np.array([[edad],[sexo],[bmi],[sys],[dia],[fuma],[padres]])
Y = np.array([[target]])



print(P.shape)
print(Y.shape)

IW, LW, b = prn.w2Wb(net)  # input-weight matrices,connection weight matrices, bias vectors

########################
# 1. Calculate NN Output
#Y_NN, n, a = prn.NNOut_(P, net, IW, LW, b, a=1, q0=0)
Y_NN = prn.NNOut(P,net)
print("Predicted: ")
print(Y_NN)
########################
# 2. Calculate Cost function E
Y_delta = Y - Y_NN  # error matrix
e = np.reshape(Y_delta, (1, np.size(Y_delta)), order='F')[0]  # error vector
E = np.dot(e, e.transpose())  # Cost function (mean squared error)

#np.savetxt("Y_NN.csv", Y_NN, delimiter=",")
#np.savetxt("Y.csv", Y, delimiter=",")

print(E)

