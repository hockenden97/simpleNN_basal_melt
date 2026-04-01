import numpy as np 
import sys
print("Welcome to this script")

seed_nb = int(sys.argv[1])

x = np.arange(0,seed_nb,1)
print(x)
np.savetxt('x.csv',x,delimiter=",")

