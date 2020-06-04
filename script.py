import numpy as np

A = np.asarray([0.75, 0.75, 0.65, 0.46, 0.38, 0.25, 0.56, 0.25, 0.19, 0.003])
B = np.asarray([0.71, 0.74, 0.63, 0.41, 0.38, 0.21, 0.51, 0.22, 0.24, 0.002])

C = (B-A)/A
s = ""
for i in range(10):
    s = s + "{:.1f}".format(C[i]*100) + '\\%, '
print(s)