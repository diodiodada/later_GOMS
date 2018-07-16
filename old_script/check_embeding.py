from keras.models import Model
from keras.layers import *
from keras.optimizers import *
import pickle
import numpy as np
from itertools import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE


def reshape_data(filename):

    data = pickle.load(open(filename, 'rb'))

    # data normalization
    x = np.array([0, 5, 8, 11, 14, 17, 20, 27, 32, 35, 38, 41, 44, 47])
    y = x + 1
    z = x + 2

    data_x = data[:, x]
    data_y = data[:, y]
    data_z = data[:, z]

    data_x_mean = 1.3020565993173778
    data_x_std = 0.1208025863782076
    data_y_mean = 0.7487616786156264
    data_y_std = 0.12072576091100165
    data_z_mean = 0.4391706194317077
    data_z_std = 0.03664239363082237

    data_x = (data_x - data_x_mean) / data_x_std
    data_y = (data_y - data_y_mean) / data_y_std
    data_z = (data_z - data_z_mean) / data_z_std

    data[:, x] = data_x
    data[:, y] = data_y
    data[:, z] = data_z

    # count
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
    print("mean:", num_length.mean(),
          "variance:", num_length.var(),
          "max:", num_length.max(),
          "min:", num_length.min(),
          "length:", num_length.shape[0])

    data_reshape = np.zeros((num_length.shape[0], 400, 51))

    for i in range(num_length.shape[0]):
        data_reshape[i, 0:num_length[i], :] = data[num_index[i] - num_length[i] + 1:num_index[i] + 1, :]

    return data_reshape


def encoder_only():
    states = Input(shape=(100, 17))

    # =========== encoder ===========

    e_dense_1 = Dense(50, activation='relu', name="e_dense_1")(states)
    e_dense_2 = Dense(50, activation='relu', name="e_dense_2")(e_dense_1)

    e_lstm_1 = LSTM(100, return_sequences=False, stateful=False, name="e_lstm_1")(e_dense_2)

    e_dense_3 = Dense(50, activation='relu', name="e_dense_3")(e_lstm_1)
    e_dense_4 = Dense(24, activation='softmax', name="e_dense_4")(e_dense_3)

    model = Model(inputs=states,
                  outputs=e_dense_4,
                  name='encoder_model')

    # print(model.summary())

    return model


def get_embeding(model):

    model.compile(optimizer=Adam(lr=1e-4),
                  loss='mse',
                  )

    model.load_weights('weights-0/1-tra-encoder.709-0.341148.hdf5', by_name=True)

    # model.reset_states()

    result_all = []

    a = [0, 1, 2, 3]
    for perm in permutations(a):
        name = str(list(perm))
        s = 'data/PP-1-paths-20-' + name + '.p'

        data = reshape_data(s)
        data = data[:, 0::4, 0:17]
        results = model.predict_on_batch(data)
        result_all.extend(list(results))

    result_all = np.array(result_all)

    pickle.dump(result_all, open("embeding/embeding_result-301.p", "wb"))

    print(result_all.shape)

    print()
    for i in range(result_all.shape[0]):
        print(result_all[i].argmax(), end=" ")
        # print(result_all[i].max(), end=" ")
        if (i+1) % 20 == 0:
            print("")

    cnames = {
        'black': '#000000',
        'blue': '#0000FF',
        'brown': '#A52A2A',
        'chocolate': '#D2691E',
        'deeppink': '#FF1493',
        'lavender': '#E6E6FA',
        'lavenderblush': '#FFF0F5',
        'magenta': '#FF00FF',
        'maroon': '#800000',
        'mediumaquamarine': '#66CDAA',
        'mediumblue': '#0000CD',
        'mediumorchid': '#BA55D3',
        'seagreen': '#2E8B57',
        'seashell': '#FFF5EE',
        'sienna': '#A0522D',
        'silver': '#C0C0C0',
        'skyblue': '#87CEEB',
        'slateblue': '#6A5ACD',
        'slategray': '#708090',
        'snow': '#FFFAFA',
        'springgreen': '#00FF7F',
        'steelblue': '#4682B4',
        'thistle': '#D8BFD8',
        'tomato': '#FF6347',
        'violet': '#EE82EE',
        'white': '#FFFFFF',
        'whitesmoke': '#F5F5F5',
        'yellow': '#FFFF00',
        'yellowgreen': '#9ACD32'}

    X = TSNE(n_components=3, init='pca', random_state=0).fit_transform(result_all)

    fig = plt.figure()

    ax = Axes3D(fig)

    for i in range(24):
        ax.scatter(X[i*20:(i+1)*20, 1], X[i*20:(i+1)*20, 0], X[i*20:(i+1)*20, 2], color=cnames[list(cnames.keys())[i]])

    plt.show()


model = encoder_only()
get_embeding(model)
