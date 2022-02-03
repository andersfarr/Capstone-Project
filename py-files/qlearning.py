import numpy as np
import random
from functions import *

def generateQtable(capacity, demand, alpha, gamma, epsilon, iterations, binSize):

    # initialize spaces
    state_space = [quantize(i, binSize) for i in range(capacity + 1)]
    quantized_state_space = sorted(list(set([quantize(i, binSize) for i in state_space])))
    action_space = [i for i in range(capacity + 1)]
    noise_space = demand

    # initialize q table
    q_table = np.zeros([len(quantized_state_space), len(action_space)])

    # initialize state
    state = random.choice(state_space)
    quantized_state = quantize(state, binSize)

    # perform iterations
    for i in range(iterations):

        # get action subspace
        action_subspace = [u for u in action_space if state + u <= capacity]

        if random.uniform(0,1) < epsilon:
            action = random.choice(action_subspace)
        else:
            action = np.argmax(q_table[quantized_state_space.index(quantized_state)])

        # get stochastic noise
        noise = random.choice(noise_space)

        # calculate profit
        profit = costFunction(state, action, noise)

        # get next state
        if state + action - noise < 0:
            next_state = 0
        elif state + action - noise > capacity:
            next_state = capacity
        else:
            next_state = state + action - noise
        
        next_quantized_state = quantize(next_state, binSize)

        # estimate optimal future value
        next_max = np.max(q_table[quantized_state_space.index(next_quantized_state)])

        # get old value
        old_value = q_table[quantized_state_space.index(quantized_state), action]

        # calculate new value
        new_value = ((1 - alpha) * old_value) + alpha * (profit + (gamma * next_max))

        # save new value
        q_table[quantized_state_space.index(quantized_state), action] = new_value

        # go to next state
        state = next_state
        quantized_state = next_quantized_state

    return q_table

def generateQpolicies(q_table):

    # turn to np array
    q_table = np.array(q_table)

    # get policies
    q_policies = q_table.argmax(axis = 1)

    return q_policies