from random import uniform, randint, gauss
import random
from math import exp
import matplotlib.pyplot as plt
import numpy as np

random_seed = 31  # Fix random seed in spin initialization
nsweeps = 1000  # Number of sweeps
sigma = 1.0  # Standard deviation for the initialisation of the lattice

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

    return action_local


def get_action_of_site(phi, x, m_squared, l, neighbours_dictionary):

    neighbours = neighbours_dictionary[x]
    left, right, top, bottom = neighbours[0], neighbours[1], neighbours[2], neighbours[3]
    action_of_site = phi[x]*(4*phi[x]-phi[left]-phi[right]-phi[bottom]-phi[top]) + m_squared*(phi[x]**2) + l*(phi[x]**4)

    return action_of_site


def get_action_difference(phi, x, m_squared, l, new_value, neighbours_dictionary):

    neighbours = neighbours_dictionary[x]
    left, right, top, bottom = neighbours[0], neighbours[1], neighbours[2], neighbours[3]

    action_local_old = get_action_of_site(phi, x, m_squared, l, neighbours_dictionary) + \
                       get_action_of_site(phi, left, m_squared, l, neighbours_dictionary) + \
                       get_action_of_site(phi, right, m_squared, l, neighbours_dictionary) + \
                       get_action_of_site(phi, top, m_squared, l, neighbours_dictionary) + \
                       get_action_of_site(phi, bottom, m_squared, l, neighbours_dictionary)

    phi_tmp = phi.copy()
    phi_tmp[x] = new_value

    action_local_new = get_action_of_site(phi_tmp, x, m_squared, l, neighbours_dictionary) + \
                       get_action_of_site(phi_tmp, left, m_squared, l, neighbours_dictionary) + \
                       get_action_of_site(phi_tmp, right, m_squared, l, neighbours_dictionary) + \
                       get_action_of_site(phi_tmp, top, m_squared, l, neighbours_dictionary) + \
                       get_action_of_site(phi_tmp, bottom, m_squared, l, neighbours_dictionary)

    return action_local_new - action_local_old


def metropolis(phi, N, m_squared, l, neighbours_dictionary):

    acc = 0

    for i in range(N):

        x = randint(0, N-1) # Pick a site at random
        new_value = gauss(phi[x], sigma)
        dS = get_action_difference(phi, x, m_squared, l, new_value, neighbours_dictionary)
        rdm_num = uniform(0, 1)

        if dS <= 0:
            acc += 1
            phi[x] = new_value

        elif exp(-dS) > rdm_num:
            acc += 1
            phi[x] = new_value

        else:
            continue

    print(acc/N)


def simulate():

    L_list = [6, 8, 10, 12, 14]
    m_squared = -4
    l_list = [6.975, 6.008, 5.550, 5.276, 5.113]
    N_list = np.array(L_list)**2
    n_sweep = list([i for i in range(nsweeps)])

    with open('phi4.txt', 'w') as file:

        for i in range(len(L_list)):

            N = N_list[i]
            l = l_list[i]
            L = L_list[i]
            random.seed(random_seed)  # Fix seed for initialisation
            phi = list([gauss(0, sigma) for _ in range(N)])  # Initialise a lattice
            neighbours_dictionary = get_neighbours_index(L)  # Initialise the dictionary holding the neighbours

            action = []

            for j in range(nsweeps):
                if j >= nsweeps/100:  # Thermalise by omitting first 1/4 and take every 10 samples
                    file.writelines("%s," % val for val in phi)
                    file.write('\n')
                metropolis(phi, N, m_squared, l, neighbours_dictionary)
                print(L)

                action.append(get_action(phi, N, m_squared, l, neighbours_dictionary))

            plt.plot(n_sweep, action)
            plt.show()


simulate()
