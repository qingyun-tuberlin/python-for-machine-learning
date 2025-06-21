import numpy as np

"""
We have a numpy array a. First construct a list of indices, name it as indices.
The content of indices should be [0,1,2,...,len(array)].
Here we can use numpy.arange(<size>) as a convinient way.
We shuffle the indices by applying numpy.random.shuffle() on the array.
At end, apply the shuffled indices on the array.
"""
a = np.array([0,10,20,30])
indices = np.arange(len(a))
np.random.shuffle(indices)
a_train_shuffled = a[indices]

print("output")
print("indices:\n",indices)
print("a_train_shuffled:\n",a_train_shuffled)

# output:
# indices:
#  [1 3 0 2]
# a_train_shuffled:
#  [10 30  0 20]