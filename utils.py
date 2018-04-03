import numpy as np
from smh import specutils

def get_order_num(order):
    """
    Classify the order into a number based on its dispersion
    """
    wave = order.dispersion
    w1, w2 = wave[0], wave[-1]
    dw = np.median(np.diff(wave))
    
def match_orders(all_orders, dwl=17.):
    order_limits = []
    all_order_labels = []
    for orders in all_orders:
        order_labels = []
        for order in orders:
            for i,order_limit in enumerate(order_limits):
                # if you find a matching order limit...
                if np.abs(order.dispersion[0]-order_limit[0]) < dwl and \
                   np.abs(order.dispersion[-1]-order_limit[1]) < dwl:
                        order_labels.append(i)
                        break
            else: # if you didn't find any limits matching...
                order_labels.append(len(order_limits))
                order_limits.append((order.dispersion[0], order.dispersion[-1]))
                
        all_order_labels.append(order_labels)
    return order_limits, all_order_labels

def round10up(x):
    if not np.isfinite(x): return x
    return 10*int(np.floor(x/10)) + 10*(x % 10 != 0)
def round10down(x):
    if not np.isfinite(x): return x
    return 10* int(np.ceil(x/10))

def flatten_list_of_lists(list_of_lists):
    flattened = []
    for l in list_of_lists:
        flattened += l
    return np.array(flattened)
    
