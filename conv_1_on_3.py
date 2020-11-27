import numpy as np


def conv_1_on_3(data_vec_np, filter_np):
    result_np = np.empty(data_vec_np.size)
    for ind in range(result_np.size):
        if ind-1 < 0 :
            left_neighboor = 0
        else:
            left_neighboor = data_vec_np[ind - 1]

        if ind+1 >= data_vec_np.size :
            right_neighboor = 0
        else:
            right_neighboor = data_vec_np[ind + 1]

        result_np[ind] = left_neighboor*filter_np[0] + data_vec_np[ind]*filter_np[1] + right_neighboor*filter_np[2]

    return result_np


# vec = np.array([-1, 1, -1.5, 1, -1.5, 1, -1, 1])
# filter_vec = np.array([1.5, 0, 0])
# print(conv_1_on_3(vec, filter_vec))


