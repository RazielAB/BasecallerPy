import numpy as np


def batch_norm(np_data_vec, beta, gamma):
    # calculate std
    mean = np_data_vec.mean()
    square_vec = np_data_vec * np_data_vec
    mean_square_vec = square_vec.mean()
    std = np.sqrt(mean_square_vec - mean * mean)
    return beta + gamma * np_data_vec / std


# np_data_vec_input = np.array([0.5, 1 , 0.5 , 1 , -3 , -5 , 4])
# print(batch_norm(np_data_vec_input, 0 , 1))