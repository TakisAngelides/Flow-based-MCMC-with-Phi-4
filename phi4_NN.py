# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
# import numpy as np

L = 5
N = int(L**2)
N_samples = 20

def get_neighbours_index(L):
    """
    :return: (dict) containing as keys the index of the site on the lattice and as values a list containing the indexes
    of its neighbours
    """
    neighbours_dict = {}  # Index of site -> indexes of neighbours of that site (key -> value)
    N = L**2
    for i in range(N):
        # store index of neighbours in the values for each node (key, i) in the lattice
        # in the form left, right, top, bottom with periodic boundary conditions
        if i % L == 0:
            left = i + L - 1
        else:
            left = i - 1
        if (i + 1) % L == 0:
            right = i - L + 1
        else:
            right = i + 1
        if i - L < 0:
            top = i - L + N
        else:
            top = i - L
        if i + L >= N:
            bottom = i + L - N
        else:
            bottom = i + L

        neighbours_dict[i] = [left, right, top, bottom]

    return neighbours_dict

def get_action(phi, N, m_squared, l, neighbours_dictionary):

    action_local = 0

    for x in range(N):

        neighbours = neighbours_dictionary[x]
        left, right, top, bottom = neighbours[0], neighbours[1], neighbours[2], neighbours[3]
        action_local += phi[x]*(4*phi[x]-phi[left]-phi[right]-phi[bottom]-phi[top]) + m_squared*(phi[x]**2) + l*(phi[x]**4)

def get_black_white_indices(L):

    black = [] # First square is black
    white = []
    N = int(L**2)

    for idx in range(N):
        row = int(idx/3)
        col = idx % 3
        if (row + col) % 2 == 0:
            black.append(idx)
        else:
            white.append(idx)

    return black, white

get_black_white_indices(6)

# def loss_function():
#     return
#
# inputs = keras.Input(shape=(N/2,))
# x1 = layers.Dense(64, activation="relu")(inputs)
# x2 = layers.Dense(64, activation="relu")(x1)
# outputs = layers.Dense(N/2)(x2)
# model = keras.Model(inputs=inputs, outputs=outputs)
#
# # Instantiate an optimizer.
# optimizer = keras.optimizers.SGD(learning_rate=1e-3)
# # Instantiate a loss function.
# loss_fn = loss_function
#
# # Prepare the training dataset.
# batches = 4
# batch_size = N_samples//batches
# X_train = np.random.normal(0, 1, (N_samples, N))
#
# epochs = 2
# for epoch in range(epochs):
#     for batch in range(batches):
#         X_input = X_train[batch*batch_size:(batch+1)*batch_size]
