import numpy as np
import pickle


max_data_length = 120
# np.random.shuffle(x)


def get_data(filename):

    data = pickle.load(open(filename, 'rb'))

    # count
    last_index = 0
    num_length = []
    num_index = []

    for i in range(data.shape[0]):
        if data[i, -1] == 1.0:
            length = i - last_index + 1
            num_length.append(length)
            num_index.append(i)
            last_index = i + 1

    num_length = np.array(num_length)
    num_index = np.array(num_index)
    print("mean:", num_length.mean(),
          "variance:", num_length.var(),
          "max:", num_length.max(),
          "min:", num_length.min(),
          "length:", num_length.shape[0])

    data_reshape = np.zeros((num_length.shape[0], max_data_length, 52))

    for i in range(num_length.shape[0]):
        data_reshape[i, 0:num_length[i], :] = data[num_index[i] - num_length[i] + 1:num_index[i] + 1, :]

    return data_reshape


data_1 = get_data("data/PP-1-paths-1000-[0]-end-flag-random-init.p")
data_2 = get_data("data/PP-1-paths-1000-[2]-end-flag-random-init.p")

print(data_1.shape)
print(data_2.shape)

data = np.concatenate((data_1, data_2), axis=0)
print(data.shape)

np.random.shuffle(data)

pickle.dump(data, open("data/PP-1-paths-2000-[0-2]-end-flag-random-init.p", "wb"))


