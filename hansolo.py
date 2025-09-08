import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

def GainOutput(K_output, K_mid, P_mid, target): 
    """
    Compute the gains of the output layer.
    """
    gain = np.zeros((K_output, K_mid))
    X = np.arange(K_output)
    out_not_ok = np.where(X != target)[0]
    gain[target, :] = 10 * P_mid
    gain[out_not_ok[:,None],:]=  - 10 * P_mid
    return gain

def in_A(obj, task):
    if task == 'easy1':
        return obj[0] == 1 
    elif task == 'easy2':
        return (obj[0] == 1 and obj[3] == 1) or (obj[0] == 1 and obj[5] == 1) or (obj[1] == 1 and obj[4] == 1) or (obj[2] == 1 and obj[3] == 1) or (obj[2] == 1 and obj[4] == 1)

#listA = [
#    [1, 0, 0, 1, 0, 0],
#    [1, 0, 0, 0, 0, 1],
#    [0, 1, 0, 0, 1, 0],
#    [0, 0, 1, 1, 0, 0],
#    [0, 0, 1, 0, 1, 0],
#] #3 shapes, 3 colors ex blue circle, blue triangle, red square, green circle, green square


def in_B(obj,task):  # everything except the blue circle
    return not in_A(obj,task)

def GainOutputEasyTask(K_output, K_mid, P_mid, obj, task): 
    """
    Compute the gains of the output layer.
    """
    gain = np.zeros((K_output, K_mid))
    X = np.arange(K_output)
    if task == 'easy1':
        propA = 1/3
    elif task == 'easy2':
        propA = 5/9
    propB = 1-propA
    if in_A(obj,task):
        gain[0, :] = (1/propA) * P_mid
        gain[1,:]=  - (1/propB)* P_mid
    else:
        gain[0, :] = -(1/propA) * P_mid
        gain[1,:]=   (1/propB) * P_mid
    return gain

def GainOutputAllConnected(K_output, K_mid, K_input, param, P_mid, P_input, target):
    """
    Compute the gains of the output layer when output neurons are connected to input neurons as well.
    """
    P_tot = np.concatenate((P_mid, P_input * param))
    K_tot = K_input + K_mid
    gain = np.zeros((K_output, K_tot))
    X = np.arange(K_output)
    out_not_ok = np.where(X != target)[0]
    gain[target, :] = 10 * P_tot
    gain[out_not_ok[:,None],:]=  - 10 * P_tot / 9
    return gain


def GainMid(mid_neur, K_mid, K_input, input_neurons, l_mid):
    """
    Compute the gains of a hidden layer.
    """
    gain = np.zeros((K_mid, K_input))
    N = input_neurons.shape[1]
    i_1 = l_mid[mid_neur, 0]
    i_2 = l_mid[mid_neur, 1]
    
    gain[mid_neur, :] = (1/N) * np.sum(input_neurons * input_neurons[i_1, None, :] * input_neurons[i_2, None, :], axis=2)
    
    return gain


def EWA(W_not_renorm, eta, cred):
    """
    Updates the weights `W_not_renorm` using the Exponentially Weighted Average (EWA) method.
    """
    res = W_not_renorm.copy()
    res[:, :] *= np.exp(eta * cred[:, :])
    return res



def PWA(p, K_output, K_input, cred_cum_output, cred_cum_input):
    """
    Updates the weights `W_not_renorm` using the PWA method.
    """
    diff = np.maximum(
        0, cred_cum_input[:K_output, :K_input] - cred_cum_output[:K_output, np.newaxis]
    )
    #res = diff ** (p - 1) para = 2
    res = diff
    return res

def list_obj_random(obj, M):
    """
    Create a matrix of random permutations of the input list 'obj' until it reaches
    the shape (M, len(obj)), where each row represents a block of randomly permuted objects. 
    Note: this function assumes that M is a multiple of n.
    """
    n = len(obj)
    num_blocks = int(M / n)
    
    # Create an array of random permutations for each block
    permutations = np.array([np.random.permutation(obj) for _ in range(num_blocks)])
    
    # Stack the permutations vertically to form the matrix
    result_matrix = np.vstack(permutations)
    
    return result_matrix

