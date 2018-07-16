import pickle
import numpy as np
import matplotlib.pyplot as plt

data_original = pickle.load(open('Pick-Place-Push-reshaped-category-1000.p', 'rb'))

for j in range(10):
    data = data_original[j, :, :]
    print(data.shape)

    x = np.linspace(0, 179, 180)
    y = np.zeros((180,))
    print(x.shape)
    print(y.shape)

    for i in range(180):
        D_1 = data[i, 5:8] - data[i, 23:26]
        D_2 = data[i, 14:17] - data[i, 23:26]

        # D_1 = abs(D_1[0]) + abs(D_1[1]) + abs(D_1[2])
        # D_2 = abs(D_2[0]) + abs(D_2[1]) + abs(D_2[2])

        D_1 = np.sqrt(np.square(D_1[0]) + np.square(D_1[1]) + np.square(D_1[2]))
        D_2 = np.sqrt(np.square(D_2[0]) + np.square(D_2[1]) + np.square(D_2[2]))

        y[i] = D_1 + D_2

    plt.figure()
    plt.plot(x, y)
    plt.show()
