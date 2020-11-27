import numpy as np

from BatchNorm import batch_norm
from conv_1_on_3 import conv_1_on_3


class ConvLayerFilters:
    def __init__ (self, first_filter, second_filter, third_filter, side_filter):
        self.first_filter  =  first_filter
        self.second_filter = second_filter
        self.third_filter  = third_filter
        self.side_filter   = side_filter


def conv_layer(input_vec, conv_filters, beta, gamma ):
    first_filtered = conv_filters.first_filter[0]*input_vec +conv_filters.first_filter[1]*input_vec
    first_filtered_bn = batch_norm(first_filtered, beta, gamma)
    first_filtered_relu = np.maximum(first_filtered_bn, 0)
    conv3_ch0 = conv_1_on_3(first_filtered_relu, conv_filters.second_filter[0])
    conv3_ch1 = conv_1_on_3(first_filtered_relu, conv_filters.second_filter[1])
    second_filtered = conv3_ch0 + conv3_ch1
    second_filtered_bn = batch_norm(second_filtered, beta, gamma)
    second_filtered_relu = np.maximum(second_filtered_bn, 0)

    third_filtered = conv_filters.third_filter[0] * second_filtered_relu + conv_filters.third_filter[1] * second_filtered_relu
    third_filtered_bn = batch_norm(third_filtered, beta, gamma)

    side_filtered = conv_filters.side_filter[0] * input_vec + conv_filters.side_filter[1] * input_vec
    side_filtered_bn = batch_norm(side_filtered, beta, gamma)

    final_add = third_filtered_bn + side_filtered_bn

    output = batch_norm(final_add, beta, gamma)
    return output

# beta  = 0
# gamma = 1
# myfilters = ConvLayerFilters(np.array([1,1]), np.array([(1,1,1),(1,1,1)]), np.array([1,1]),np.array([1,1]) )
# print(conv_layer(np.array([1,2,3,4,5,6,7,8]), myfilters,beta, gamma))
