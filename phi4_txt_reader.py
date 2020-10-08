import numpy as np
import matplotlib.pyplot as plt

with open("phi4.txt", "r") as file:

    configurations = []
    lengths = []

    for line in file.readlines():

        row = line.strip().split(',')
        lengths.append(len(row)-1)  # -1 because there is an empty last element to discard
        configurations.append(list(map(float, row[:-1])))

    N_list = list(set(lengths))
    N_list = sorted(N_list)
    num_of_L = len(N_list)
    L_list = np.array(N_list)**0.5
    nsweeps = int(len(configurations)/len(L_list))

def get_C_constant(configs, L):

    configs = np.array(configs)
    return np.sum(np.sum(configs))/(nsweeps*L*L)

def get_chi2(configs, L):

    configs = np.array([np.array(configs[blah]).reshape(L,L) for blah in range(len(configs))])
    term1_matrices = []
    for x_shift in range(L):
        for y_shift in range(L):
            shiftx_configs = np.array([np.roll(configs[i],x_shift,axis=1) for i in range(len(configs))]) # shift in x
            shiftxy_configs = np.array([np.roll(shiftx_configs[j], y_shift, axis=0) for j in range(len(shiftx_configs))]) # shift in y
            term1_matrices.append(configs*shiftxy_configs) # matrix multiplication of the first term in G for all x0,y0
    term1 = np.array([np.sum(np.array(term1_matrices[k])) for k in range(len(term1_matrices))]) # values with (t=0,t=1,t=2,...)

    return np.sum(term1)/nsweeps/L**2 # average over configurations

def get_chi2_t(configs, L):

    C = get_C_constant(configs, L)
    configs = np.array([np.array(configs[m]).reshape(L,L) for m in range(len(configs))])
    term1_matrices = []

    for x_shift in range(L):
        for y_shift in range(L):
            shiftx_configs = np.array([np.roll(configs[i],x_shift,axis=1) for i in range(len(configs))]) # shift in x
            shiftxy_configs = np.array([np.roll(shiftx_configs[j], y_shift, axis=0) for j in range(len(shiftx_configs))]) # shift in y
            term1_matrices.append((configs-C)*(shiftxy_configs-C)) # matrix multiplication of the first term in G for all x0,y0

    # shape of term1_matrices (L^2,nsweeps,L,L)
    # sum over the first dimension by adding the matrices element wise (first dimension coming from the different ways
    # of shifting)

    sum_vector = np.zeros((np.shape(term1_matrices)[1],np.shape(term1_matrices)[2],np.shape(term1_matrices)[3]))
    for dim1 in range(len(term1_matrices)):
        sum_vector += term1_matrices[dim1]

    # Convert the final nsweep matrices into numbers by summing their elements up for each matrix
    # the scalar vector will end up being of the form: values at (t=0, t=1, t=2, ... , t=tmax=nsweep)

    scalar_vector = []
    for n in range(np.shape(term1_matrices)[1]):
        scalar_vector.append(np.sum(sum_vector[n]))

    return np.array(scalar_vector)/L**2

def get_chi2_giannis(configs, L):
    y0,x0 = np.mgrid[:L,:L]
    v00 = (x0 + y0*L).flatten()
    p = np.array(configs).reshape([-1, L**2])
    N = p.shape[0]
    C0 = np.zeros([N, L**2])
    for y in range(L):
        for x in range(L):
            vyx = ((x0 + x)%L + ((y0 + y)%L)*L).flatten()
            C0[:, y*L + x] = (p[:, v00]*p[:, vyx]).sum(axis=1)
    return np.sum(C0)/N/L**2

data = {}

for j in range(len(L_list)):
    L = int(L_list[j])
    N = int(L**2)
    configs = configurations[j*nsweeps:(j+1)*nsweeps]
    data[N] = list(get_chi2_t(configs, L))

def autocorrelation(obs_list, tau):

    tmax = len(obs_list)
    c = 1 / (tmax - tau)
    sum1 = 0
    sum2 = 0

    for t in range(tmax-tau):
        sum1 += obs_list[t]*obs_list[t+tau]
        sum2 += obs_list[t]

    return c*sum1-(c**2)*(sum2**2)

data_autocorrelation = {}

for N in N_list:
    obs_list = data[N]
    data_autocorrelation[N] = [autocorrelation(obs_list, tau) for tau in range(nsweeps//10)]

def tau_int(autocorrelation_list):

    distance_list = [abs(autocorrelation_list[i]/autocorrelation_list[0]-1*(1/np.e)) for i in range(len(autocorrelation_list))]

    return distance_list.index(min(distance_list))

tau_int_list = []

for N in N_list:
    tau_int_list.append(tau_int(data_autocorrelation[N]))

plt.scatter(L_list, tau_int_list, color = 'k', marker = 'x')
plt.yscale('log')
plt.show()

tau_values = [tt for tt in range(nsweeps//10)]

plt.plot(tau_values, np.array(data_autocorrelation[N_list[0]])/(data_autocorrelation[N_list[0]][0]))
plt.vlines(tau_int_list[0],ymin=0,ymax=1)
plt.show()
