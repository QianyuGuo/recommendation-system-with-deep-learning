import numpy as np
import pandas as pd
def vectorize_dic(dic,ix=None,p=None,n=0,g=0):
    """
    dic -- dictionary of feature lists. Keys are the name of features
    ix -- index generator (default None)
    p -- dimension of featrure space (number of columns in the sparse matrix) (default None)
    """
    if ix==None:
        ix = dict()

    nz = n * g

# numpy.empty(shape, dtype=float, order='C')
# Return a new array of given shape and type, without initializing entries.

# Parameters: 
# shape : int or tuple of int

# Shape of the empty array

# dtype : data-type, optional

# Desired output data-type.

# order : {‘C’, ‘F’}, optional

# Whether to store multi-dimensional data in row-major (C-style) or column-major (Fortran-style) order in memory.

# Returns:    
# out : ndarray

# Array of uninitialized (arbitrary) data of the given shape, dtype, and order. Object arrays will be initialized to None.
# >>> np.empty([2, 2])
# array([[ -9.74499359e+001,   6.69583040e-309],
#        [  2.13182611e-314,   3.06959433e-309]])         #random
# >>> np.empty([2, 2], dtype=int)
# array([[-1073741821, -1067949133],
#        [  496041986,    19249760]])                     #random
    col_ix = np.empty(nz,dtype = int)

    i = 0
    for k,lis in dic.items():
        for t in range(len(lis)):
            ix[str(lis[t]) + str(k)] = ix.get(str(lis[t]) + str(k),0) + 1
            col_ix[i+t*g] = ix[str(lis[t]) + str(k)]
        i += 1

    row_ix = np.repeat(np.arange(0,n),g)
    data = np.ones(nz)
    if p == None:
        p = len(ix)

    ixx = np.where(col_ix < p)
    return csr.csr_matrix((data[ixx],(row_ix[ixx],col_ix[ixx])),shape=(n,p)),ix

cols = ['user','item','rating','timestamp']

train = pd.read_csv('data/ua.base',delimiter='\t',names = cols)
test = pd.read_csv('data/ua.test',delimiter='\t',names = cols)

x_train,ix = vectorize_dic({'users':train['user'].values,
                            'items':train['item'].values},n=len(train.index),g=2)


x_test,ix = vectorize_dic({'users':test['user'].values,
                           'items':test['item'].values},ix,x_train.shape[1],n=len(test.index),g=2)


y_train = train['rating'].values
y_test = test['rating'].values

x_train = x_train.todense()
x_test = x_test.todense()