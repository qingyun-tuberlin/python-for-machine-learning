# numpy has magic :D

# we have a list of probalilities. The list contains 
# N samples and its probabiliy distribution of items.
# We also have a list of labels, the labels contains the
# index of item in the sample, which represents where is
# the largest probabiliy.
# The task is to extract those largest probalilities by using 
# numpy

# For better understanding, 
# also solve the task by using pure python, please.

import numpy as np

# do the task with pure python
probalilities = [
    [0.01, 0.7, 0.02],  # sample 0
    [0.3, 0.4, 0.7], # sample 1
    [0.8, 0.1, 0.2]  # sample 2
]

# the truth label of each sample
# the label[i] means the index of the correct class
labels = [1,2,0]
correct_probabilities = []

for i in range(len(probalilities)):
    label = labels[i]
    probabiliy = probalilities[i][label]
    correct_probabilities.append(probabiliy)

print("correct_probabilities: \n",correct_probabilities)

# do the task with numpy
probs = np.array(probalilities)
length = len(probs)
tags = np.array(labels)
correct_probs = probs[range(length),tags]
print("correct_probs:\n",correct_probs)