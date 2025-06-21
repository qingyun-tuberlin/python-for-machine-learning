# argmax takes the index of the item with the largest value
# if the array's size is (n,), then argmax returns the index of the item
# with the biggest value 
# if the array's size is (n,m), then argmax returns a list of indices, which
# represent the index of the largest item's value
# the parameter axis specifies if the array is handled colum wise or row wise

import numpy as np

output = np.array([
    [0.1,0.8,0.1],
    [0.2,0.1,0.7],
    [0.33,0.9,0.7]
])

# axis	Meaning	        Acts on
# 0	    Column-wise	    Each column
# 1	    Row-wise	    Each row
pred_colwise = np.argmax(output, axis=0)
print("pred_colwise:\n",pred_colwise)

pred_rowwise = np.argmax(output, axis=1)
print("pred_colwise:\n",pred_rowwise)