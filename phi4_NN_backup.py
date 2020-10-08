import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense
import numpy as np
from cmath import exp

L_list = [6, 8, 10, 12, 14]
m_squared = -4
l_list = [6.975, 6.008, 5.550, 5.276, 5.113]
N_list = np.array(L_list) ** 2
N_samples = 10
N_configs = 10


def get_action(phi, N, m_squared, l):
    samples = phi.shape[0]
    L = int(N ** 0.5)

    phi = tf.reshape(phi, (samples, L, L))
    phi_left = tf.roll(phi, shift=-1, axis=1)
    phi_right = tf.roll(phi, shift=1, axis=1)
    phi_top = tf.roll(phi, shift=-1, axis=0)
    phi_bottom = tf.roll(phi, shift=1, axis=0)

    term0 = tf.math.add(tf.reshape(4 * phi, (samples, N)), tf.reshape(-phi_left, (samples, N)))
    term00 = tf.math.add(tf.reshape(term0, (samples, N)), tf.reshape(-phi_right, (samples, N)))
    term000 = tf.math.add(tf.reshape(term00, (samples, N)), tf.reshape(-phi_top, (samples, N)))
    term1 = tf.math.add(tf.reshape(term000, (samples, N)), tf.reshape(-phi_bottom, (samples, N)))  # bs, N
    phi_squared = tf.math.square(phi)
    term2 = m_squared * phi_squared  # bs, N
    term3 = l * tf.math.square(phi_squared)  # bs, N

    sum_term4 = tf.zeros(N)
    for sample in range(samples):
        sum_term4 = tf.math.add(sum_term4, tf.multiply(tf.reshape(phi, (samples, N))[sample], term1[sample]))  # N, 1

    sum1 = tf.math.reduce_sum(sum_term4)

    sum2 = tf.zeros(1)
    for sample in range(samples):
        sum2 = tf.math.reduce_sum(term2[sample])

    sum3 = tf.zeros(1)
    for sample in range(samples):
        sum3 = tf.math.reduce_sum(term3[sample])

    return tf.math.add(sum1, sum2, sum3)


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


class Layer(keras.Model):

    def __init__(self, N):

        super(Layer, self).__init__()
        self.even = True  # Start the layer object with the even being the a subscript in the paper
        self.optimizer = keras.optimizers.Adam()

        self.model_s_odd = keras.Sequential()
        self.model_s_odd.add(Dense(64, activation='relu', input_dim=N // 2))
        self.model_s_odd.add(Dense(64, activation='relu'))
        self.model_s_odd.add(Dense(N // 2))

        self.model_s_even = keras.Sequential()
        self.model_s_even.add(Dense(64, activation='relu', input_dim=N // 2))
        self.model_s_even.add(Dense(64, activation='relu'))
        self.model_s_even.add(Dense(N // 2))

        self.model_t_odd = keras.Sequential()
        self.model_t_odd.add(Dense(64, activation='relu', input_dim=N // 2))
        self.model_t_odd.add(Dense(64, activation='relu'))
        self.model_t_odd.add(Dense(N // 2))

        self.model_t_even = keras.Sequential()
        self.model_t_even.add(Dense(64, activation='relu', input_dim=N // 2))
        self.model_t_even.add(Dense(64, activation='relu'))
        self.model_t_even.add(Dense(N // 2))

    def g(self, phi):

        z_arr = z.numpy()
        N = np.shape(z_arr)[1]
        L = int(N ** 0.5)
        black, white = get_black_white_indices(L)
        phi_even = tf.gather(phi, black, axis=1)  # samples, N/2
        phi_odd = tf.gather(phi, white, axis=1)  # samples, N/2
        if self.even:
            s = self.model_s_even(phi_even)
            t = self.model_t_even(phi_even)
            z_even = phi_even
            z_odd = (phi_odd - t) * tf.math.exp(-s)
        else:
            s = self.model_s_odd(phi_odd)
            t = self.model_t_odd(phi_odd)
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
            s = self.model_s_even(z_even)
            t = self.model_t_even(z_even)
            phi_even = z_even
            phi_odd = (z_odd - t) * tf.math.exp(-s)
        else:
            s = self.model_s_odd(z_odd)
            t = self.model_t_odd(z_odd)
            phi_even = (z_even - t) * tf.math.exp(-s)
            phi_odd = z_odd

        black = [[[i, b] for b in black] for i in range(N_samples)]
        white = [[[j, w] for w in white] for j in range(N_samples)]

        shape = tf.constant([N_samples, N])

        phi_even = tf.scatter_nd(black, phi_even, shape)
        phi_odd = tf.scatter_nd(white, phi_odd, shape)

        # phi = tf.math.add(phi_even,phi_odd)

        return phi_even, phi_odd

    def log_det_jacobian(self, phi):

        z_arr = z.numpy()
        N = np.shape(z_arr)[1]
        L = int(N ** 0.5)
        black, white = get_black_white_indices(L)
        phi_even = tf.gather(phi, black, axis=1)  # samples, N/2
        phi_odd = tf.gather(phi, white, axis=1)  # samples, N/2
        if self.even:
            return tf.math.reduce_sum(self.model_s_even(phi_even))
        else:
            return tf.math.reduce_sum(self.model_s_odd(phi_odd))


data = {}
data_log_prob = {}
for i in range(len(N_list)):

    N = N_list[i]
    l = l_list[i]
    L = L_list[i]
    layer = Layer(N)
    z = tf.random.normal((N_samples, N), mean=0.0, stddev=1.0)
    with tf.GradientTape() as tape:

        phi_even, phi_odd = layer.g_inv(z)
        phi = tf.math.add(phi_even, phi_odd)  # samples, N
        term1 = (-1 / 2) * tf.math.reduce_sum(tf.math.square(z)) / N_samples  # scalar
        term2 = layer.log_det_jacobian(phi) / N_samples  # scalar
        term3 = get_action(phi, N, m_squared, l) / N_samples
        loss_1_2 = tf.math.add(term1, term2)
        loss = tf.math.add(loss_1_2, -term3)

    mse = layer.model_s_even
    mso = layer.model_s_odd
    mte = layer.model_t_even
    mto = layer.model_t_odd
    weights = mse.trainable_weights + mte.trainable_weights  # + mto.trainable_weights + mso.trainable_weights
    grads = tape.gradient(loss, weights)
    layer.optimizer.apply_gradients(zip(grads, weights))

    z_input = tf.random.normal((N_configs, N), mean=0.0, stddev=1.0)
    phi_even, phi_odd = layer.g_inv(z_input)
    phi = tf.math.add(phi_even, phi_odd)  # N_configs, N
    phi_list = phi.numpy().tolist()

    black, white = get_black_white_indices(L)
    z_tmp = z_input.numpy()
    phi_even_tmp = tf.gather(phi, black, axis=1)  # samples, N/2
    phi_odd_tmp = tf.gather(phi, white, axis=1)
    if layer.even:
        phi_s = layer.model_s_even(phi_even_tmp)
    else:
        phi_s = layer.model_s_odd(phi_odd_tmp)
    phi_s_tmp = phi_s.numpy()
    list_tmp = []
    for sample in range(len(z_tmp)):
        log_p = np.sum((-1 / 2) * z_tmp[sample] ** 2) + np.sum(phi_s_tmp[sample])
        list_tmp.append(log_p)
    data_log_prob[i] = list_tmp

    data[i] = phi_list
    print(data_log_prob[0])

# with open('phi4_NN.txt', 'w') as file:
#     for key in data:
#         for i in range(data[key]-1):

#             configuration = data[key][i]
#             next_configuration = data[key][i+1]
#             p_estimated = p_estimated_list[i]  # TODO: keep track of this list make dictionary key is N list is p for each sample
#             next_p_estimated = p_estimated_list[i+1]

#             if acc == 1:

#                 file.writelines("%s," % val for val in phi)
#                 file.write('\n')
