
import pandas as pd
import numpy as np

homout = pd.read_csv('output/heoutput.csv', header=None).append(pd.read_csv('output/homoutput.csv', header=None))
homout = homout.transpose()
homout.columns = ['Heterogeneous', 'Homogeneous']
homout = homout.astype(np.float64)
print(homout.describe().to_latex())
