import tensorflow as tf
from tensorflow import keras, initializers
from tensorflow.keras.layers import Dense
import numpy as np
from cmath import exp
from random import uniform
import sys
import math
from math import pi
import matplotlib.pyplot as plt
np.set_printoptions(threshold=sys.maxsize)


L_list = [6, 8, 10, 12, 14]
m_squared = -4
l_list = [6.975, 6.008, 5.550, 5.276, 5.113]
N_list = np.array(L_list) ** 2
N_samples = 1024
training_iterations = 500
N_layers = 8

# get_neighbours_index and get_action_sample are for checking that that get_action is correct
def get_neighbours_index(L):
    """
    :return: (dict) containing as keys the index of the site on the lattice and as values a list containing the indexes
    of its neighbours
    """
    neighbours_dict = {}  # Index of site -> indexes of neighbours of that site (key -> value)
    N = L ** 2
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

def get_action_sample(phi, N, m_squared, l, neighbours_dictionary):
    action_local = 0

    for x in range(N):
        neighbours = neighbours_dictionary[x]
        left, right, top, bottom = neighbours[0], neighbours[1], neighbours[2], neighbours[3]
        action_local += phi[x] * (4 * phi[x] - phi[left] - phi[right] - phi[bottom] - phi[top]) + m_squared * (
                phi[x] ** 2) + l * (phi[x] ** 4)

    return action_local

def get_action(phi, N, m_squared, l):

    samples = phi.shape[0]
    L = int(N ** 0.5)

    phi = tf.reshape(phi, (samples, L, L))
    phi_left = tf.roll(phi, shift=-1, axis=2)
    phi_right = tf.roll(phi, shift=1, axis=2)
    phi_top = tf.roll(phi, shift=-1, axis=1)
    phi_bottom = tf.roll(phi, shift=1, axis=1)

    term0 = tf.math.add(tf.reshape(4 * phi, (samples, N)), tf.reshape(-phi_left, (samples, N)))
    term00 = tf.math.add(tf.reshape(term0, (samples, N)), tf.reshape(-phi_right, (samples, N)))
    term000 = tf.math.add(tf.reshape(term00, (samples, N)), tf.reshape(-phi_top, (samples, N)))
    term1_tmp = tf.math.add(tf.reshape(term000, (samples, N)), tf.reshape(-phi_bottom, (samples, N)))
    phi_squared = tf.math.square(phi)
    term2_tmp = m_squared * phi_squared  # samples, L, L
    term3_tmp = l * tf.math.square(phi_squared)  # samples, L, L

    term1_f = tf.multiply(tf.reshape(phi, (samples, N)), term1_tmp)

    # sum1 = tf.math.reduce_sum(term1_f)
    #
    # sum2 = tf.math.reduce_sum(term2_tmp)
    #
    # sum3 = tf.math.reduce_sum(term3_tmp)

    term_sum = tf.math.add(term1_f,tf.math.add(tf.reshape(term2_tmp,(samples,N)),tf.reshape(term3_tmp,(samples,N))))
    final = tf.reduce_sum(term_sum, axis = 1)

    return final

def get_black_white_indices(L):
    black = []  # First square is black
    white = []
    N = int(L ** 2)

    for idx in range(N):
        row = int(idx / 3)
        col = idx % 3
        if (row + col) % 2 == 0:
            black.append(idx)
        else:
            white.append(idx)

    return black, white

def MCMC(configurations, probabilities, action_list, m_squared, l):

    # configurations is N_samples, N
    # probabilities is 1, N_samples and holds the log of p tilde for each sample

    state_idx = 0
    N = len(configurations[0])
    L = int(N ** 0.5)
    acc = 1
    accepted_configurations = [configurations[state_idx]]

    for i in range(1, len(configurations) - 1):

        next_state_idx = i
        dS = action_list[next_state_idx]-action_list[state_idx]
        dprob = probabilities[state_idx] - probabilities[next_state_idx] # log of p tilde

        if dS - dprob <= 0:
            acc += 1
            state_idx = i
            accepted_configurations.append(configurations[next_state_idx])
            continue

        weight = exp(-dS+dprob).real
        rdm_num = uniform(0, 1)

        if weight > rdm_num:
            acc += 1
            state_idx = i
            accepted_configurations.append(configurations[next_state_idx])
            continue

    acceptance_rate = acc / len(configurations)

    return accepted_configurations, acceptance_rate

class Layer(keras.Model):

    def __init__(self, N, even_bool):

        super(Layer, self).__init__()
        self.even = even_bool
        self.neurons = 128

        self.model_s = keras.Sequential()
        self.model_s.add(Dense(self.neurons, kernel_initializer=initializers.RandomNormal(stddev=0.01) , activation=tf.nn.leaky_relu, use_bias=False, input_dim=N // 2))
        self.model_s.add(Dense(self.neurons, kernel_initializer=initializers.RandomNormal(stddev=0.01) , activation=tf.nn.leaky_relu, use_bias=False))
        self.model_s.add(Dense(N // 2, kernel_initializer=initializers.RandomNormal(stddev=0.01)))

        self.model_t = keras.Sequential()
        self.model_t.add(Dense(self.neurons, kernel_initializer=initializers.RandomNormal(stddev=0.01) , activation=tf.nn.leaky_relu, use_bias=False, input_dim=N // 2))
        self.model_t.add(Dense(self.neurons, kernel_initializer=initializers.RandomNormal(stddev=0.01) , activation=tf.nn.leaky_relu, use_bias=False))
        self.model_t.add(Dense(N // 2, kernel_initializer=initializers.RandomNormal(stddev=0.01)))

    def g(self, phi):

        phi_arr = phi.numpy()
        N = np.shape(phi_arr)[1]
        L = int(N ** 0.5)
        black, white = get_black_white_indices(L)
        phi_even = tf.gather(phi, black, axis=1)  # samples, N/2
        phi_odd = tf.gather(phi, white, axis=1)  # samples, N/2
        if self.even:
            s = self.model_s(phi_even)
            t = self.model_t(phi_even)
            z_even = phi_even
            z_odd = (phi_odd - t) * tf.math.exp(-s)
        else:
            s = self.model_s(phi_odd)
            t = self.model_t(phi_odd)
            z_even = (phi_even - t) * tf.math.exp(-s)
            z_odd = phi_odd

        black = [[[i, b] for b in black] for i in range(N_samples)]
        white = [[[j, w] for w in white] for j in range(N_samples)]

        shape = tf.constant([N_samples, N])

        z_even = tf.scatter_nd(black, z_even, shape)
        z_odd = tf.scatter_nd(white, z_odd, shape)

        z = tf.math.add(z_even, z_odd)

        return z

    def g_inv(self, z):

        z_arr = z.numpy()
        N = np.shape(z_arr)[1]
        L = int(N ** 0.5)
        black, white = get_black_white_indices(L)
        z_even = tf.gather(z, black, axis=1)  # samples, N/2
        z_odd = tf.gather(z, white, axis=1)  # samples, N/2

        if self.even:
            s = self.model_s(z_even)
            t = self.model_t(z_even)
            phi_even = z_even
            phi_odd = (z_odd - t) * tf.math.exp(-s)
        else:
            s = self.model_s(z_odd)
            t = self.model_t(z_odd)
            phi_even = (z_even - t) * tf.math.exp(-s)
            phi_odd = z_odd

        black = [[[i, b] for b in black] for i in range(N_samples)]
        white = [[[j, w] for w in white] for j in range(N_samples)]

        shape = tf.constant([N_samples, N])

        phi_even = tf.scatter_nd(black, phi_even, shape)
        phi_odd = tf.scatter_nd(white, phi_odd, shape)

        # Adding the two will bring the chessboard pattern back together without affecting the values since 0
        # is the identity element of the group under addition
        phi = tf.math.add(phi_even,phi_odd)

        return phi

    def log_det_jacobian(self, phi):

        phi_arr = phi.numpy()
        N = np.shape(phi_arr)[1]
        L = int(N ** 0.5)
        black, white = get_black_white_indices(L)
        phi_even = tf.gather(phi, black, axis=1)  # samples, N/2
        phi_odd = tf.gather(phi, white, axis=1)  # samples, N/2
        if self.even:
            return self.model_s(phi_even)
        else:
            return self.model_s(phi_odd)

data = {}
data_log_prob = {}
optimizer = keras.optimizers.Adam()

for i in range(len(N_list)): #  For each lattice size N = (6,8,10,12,14)

    N = N_list[i]
    l = l_list[i]
    L = L_list[i]
    black, white = get_black_white_indices(L)

    acc_rate_list = []
    loss_list = []
    train_iter_list = []

    for train_iter in range(training_iterations): # Train until acceptance rate reaches 0.5

        with tf.GradientTape() as tape:

            # Create a list of Layer objects that act on even and odd starting with even and ending with odd
            flag = False
            layer_list = []
            for _ in range(N_layers):
                layer_list.append(Layer(N, even_bool=not flag))
                flag = not flag

            stddev = 1.0  # Standard deviation of the prior normal z distribution
            z = (stddev*tf.sqrt(2*pi)) * tf.random.normal((N_samples, N), mean=0.0, stddev=stddev)
            term1 = (-1 / 2) * tf.math.reduce_sum(tf.math.square(z)) - ((L * L) // 2) * tf.math.log(2 * math.pi)  # scalar
            previous_state = z
            final_loss = tf.zeros(1)
            enum = 0

            list_tmp = []
            layer_term_log_p = np.zeros(N_samples)
            layer_term_log_p += (tf.reduce_sum((-1 / 2) * z ** 2, axis=1)).numpy() - ((L * L) // 2) * np.log(2 * math.pi)

            for layer in layer_list:

                next_state = layer.g_inv(previous_state)
                # v = next_state.numpy()
                if math.inf in next_state.numpy():
                    print(True, enum, train_iter)
                s_phi = layer.log_det_jacobian(next_state)  # samples, N/2
                layer_term_log_p += (tf.math.reduce_sum(s_phi, axis=1)).numpy()
                term2 = tf.math.reduce_sum(s_phi)
                final_loss = tf.math.add(final_loss, term2)
                previous_state = next_state
                enum += 1

            list_tmp = list(layer_term_log_p)

            final_loss = tf.math.reduce_sum(final_loss)
            final_loss = tf.math.add(final_loss, term1)
            action_tensor = get_action(next_state, N, m_squared, l)   # scalar
            term3 = tf.math.reduce_sum(action_tensor)
            action_list = action_tensor.numpy()
            final_loss = tf.math.add(final_loss, term3) / N_samples
            # print(term1.numpy(), term2.numpy(), term3.numpy(), final_loss.numpy())

        weights = []
        for layer in layer_list:
            ms = layer.model_s
            mt = layer.model_t
            weights = weights + ms.trainable_weights + mt.trainable_weights

        grads = tape.gradient(final_loss, weights)
        optimizer.apply_gradients(zip(grads, weights))

        # Since i will be the same for each training iteration the once it finishes training for a given N value
        # it will store the ensemble and the log probability for each sample in the ensemble and move to the next i
        phi_list = next_state.numpy().tolist()
        data_log_prob[i] = list_tmp  # Stores in index i which represents the N value the log of the probability of predicting a given final phi
        data[i] = phi_list  # Stores in index i the ensemble for a given value of N

        if train_iter % 1 == 0:
            configurations = data[i]
            probabilities = data_log_prob[i]
            config_list, acc_rate = MCMC(configurations, probabilities, action_list, m_squared, l)
            train_iter_list.append(train_iter)
            acc_rate_list.append(acc_rate)
            loss_list.append(final_loss.numpy())
            print(f'Acc rate, Loss, Train iter, L: {acc_rate, final_loss.numpy(), train_iter, L}')

    plt.plot(train_iter_list, acc_rate_list)
    plt.show()
    plt.plot(train_iter_list, loss_list)
    plt.show()
