import numpy as np
#from conv_layer import batch_norm
from conv_layer import conv_layer, ConvLayerFilters


def TopConv(input_vec, conv_filters_lst, beta, gamma ):
    result_vec = input_vec
    for stage in range(5):
        result_vec = conv_layer(result_vec, conv_filters_lst[stage], beta, gamma )

    return result_vec


conv_filters_lst = []
for stage in range(5):
    conv_filters_lst.append( ConvLayerFilters(np.array([1,1]), np.array([(1,1,1),(1,1,1)]), np.array([1,1]),np.array([1,1]) ))

print(TopConv(np.array([1,2,3,4,5,6,7,8]) , conv_filters_lst, 0 ,1  ))