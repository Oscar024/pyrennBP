import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyrenn as prn
import math


datosIn = pd.read_csv("pacientes200.csv", sep=',')




Entradas = np.array(datosIn)


dato=1




X_test = Entradas[0:7,dato-1:dato]

X_test = np.array(X_test)
datos_test=X_test.reshape(1,7)
print("Datos tested: ")
print(datos_test)
index =X_test.item((2, 0))




filename = "nnBP.csv"
net=prn.loadNN(filename)

edad =X_test.item((0, 0))
sexo = X_test.item((1, 0))
bmi = X_test.item((2, 0))
sys = X_test.item((3, 0))
dia = X_test.item((4, 0))
fuma = X_test.item((5, 0))
padres = X_test.item((6, 0))
agedb = edad*dia



target=1-math.exp(-math.exp((math.log(4) - (22.949536 + (-0.156412*edad )+( -0.202933*sexo) + (-0.033881*bmi) + (-0.05933*sys) + (-0.128468*dia) + (-0.190731*fuma) +  (-0.166121*padres) + (0.001624*agedb)))/0.876925))
riesgo = target*100
print("Target: ")
print(riesgo)


P = np.array([[edad],[sexo],[bmi],[sys],[dia],[fuma],[padres]])
Y = np.array([[target]])





IW, LW, b = prn.w2Wb(net)  # input-weight matrices,connection weight matrices, bias vectors

########################
# 1. Calculate NN Output
#Y_NN, n, a = prn.NNOut_(P, net, IW, LW, b, a=1, q0=0)
Y_NN = prn.NNOut(P,net)

print("Predicted: ")
print(Y_NN*100)

########################
# 2. Calculate Cost function E
Y_delta = Y - Y_NN  # error matrix
e = np.reshape(Y_delta, (1, np.size(Y_delta)), order='F')[0]  # error vector
E = np.dot(e, e.transpose())  # Cost function (mean squared error)

#np.savetxt("P.csv", P, delimiter=",")
#np.savetxt("Y_NN.csv", Y_NN, delimiter=",")
#np.savetxt("Y.csv", Y, delimiter=",")
print("Error mse: ")
print(E)

