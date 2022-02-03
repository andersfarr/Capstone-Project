from functions import *
from qlearning import *
import matplotlib.pyplot as plt

capacity = 200
alpha = 0.1
gamma = 0.5
epsilon = 1
iterations = 1000000
binSize = 10
mu, sigma = 100, 30
demand = np.round(np.random.normal(mu, sigma, 1000)).astype(np.int32)

q_table = generateQtable(capacity, demand, alpha, gamma, epsilon, iterations, binSize)

q_policies = generateQpolicies(q_table)

plt.plot(range(capacity+1), q_policies)