import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import tensorflow as tf
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
    return csr_matrix((data[ixx],(row_ix[ixx],col_ix[ixx])),shape=(n,p)),ix

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


#估计值计算
n,p = x_train.shape

k = 10

x = tf.placeholder('float',[None,p])

y = tf.placeholder('float',[None,1])

w0 = tf.Variable(tf.zeros([1]))
w = tf.Variable(tf.zeros([p]))

v = tf.Variable(tf.random_normal([k,p],mean=0,stddev=0.01))

#y_hat = tf.Variable(tf.zeros([n,1]))

linear_terms = tf.add(w0,tf.reduce_sum(tf.multiply(w,x),1,keep_dims=True)) # n * 1
pair_interactions = 0.5 * tf.reduce_sum(
    tf.subtract(
        tf.pow(
            tf.matmul(x,tf.transpose(v)),2),
        tf.matmul(tf.pow(x,2),tf.transpose(tf.pow(v,2)))
    ),axis = 1 , keep_dims=True)


y_hat = tf.add(linear_terms,pair_interactions)


#定义损失函数
lambda_w = tf.constant(0.001,name='lambda_w')
lambda_v = tf.constant(0.001,name='lambda_v')

l2_norm = tf.reduce_sum(
    tf.add(
        tf.multiply(lambda_w,tf.pow(w,2)),
        tf.multiply(lambda_v,tf.pow(v,2))
    )
)

error = tf.reduce_mean(tf.square(y-y_hat))
loss = tf.add(error,l2_norm)


train_op = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

#训练模型
epochs = 10
batch_size = 1000

# Launch the graph
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    for epoch in tqdm(range(epochs), unit='epoch'):
        perm = np.random.permutation(x_train.shape[0])
        # iterate over batches
        for bX, bY in batcher(x_train[perm], y_train[perm], batch_size):
            _,t = sess.run([train_op,loss], feed_dict={x: bX.reshape(-1, p), y: bY.reshape(-1, 1)})
            print(t)


    errors = []
    for bX, bY in batcher(x_test, y_test):
        errors.append(sess.run(error, feed_dict={x: bX.reshape(-1, p), y: bY.reshape(-1, 1)}))
        print(errors)
    RMSE = np.sqrt(np.array(errors).mean())
    print (RMSE)