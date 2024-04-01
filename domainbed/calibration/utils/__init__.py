import numpy
from typing import Tuple

def get_maxprob_and_onehot(probs: numpy.ndarray, labels: numpy.ndarray) -> Tuple[numpy.ndarray, numpy.ndarray]:

    maxprob_list = []
    idx_list = []

    for i in range(len(probs)):
        maxprob_list.append(numpy.max(probs[i]))
        idx_list.append(numpy.argmax(probs[i]))

    maxprob_list = numpy.array(maxprob_list)
    idx_list = numpy.array(idx_list)
    one_hot_labels = labels == idx_list

    return  maxprob_list, one_hot_labels