import pickle
import numpy as np
data = pickle.load(open('Pick-Place-Push-category-1000.p', 'rb'))

num_trajectory = 0

last_index = 0

num_length = []
num_index = []
for i in range(data.shape[0]):
    if data[i, -1] == 1.0:

        num_trajectory += 1
        length = i - last_index + 1
        num_length.append(length)
        num_index.append(i)

        last_index = i + 1

num_length = np.array(num_length)
num_index = np.array(num_index)
print("mean:", num_length.mean(),"variance:", num_length.var(), "max:",num_length.max(), "min:",num_length.min())

data_reshape = np.zeros((1000,180,69))

for i in range(1000):
    data_reshape[i,0:num_length[i],:] = data[ num_index[i]-num_length[i]+1:num_index[i]+1 , :]


# check data_reshape's value
for i in range(1000):
    print(data_reshape[i,num_length[i]-1,-1])


pickle.dump(data_reshape, open("Pick-Place-Push-reshaped-category-1000.p", "wb"))










