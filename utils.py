import numpy as np
from smh import specutils
from scipy.ndimage.filters import percentile_filter

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
    
def find_minrange_maxrange(list_of_data):
    """ Finds the minimum and maximum range """
    mindata = map(min, list_of_data)
    maxdata = map(max, list_of_data)
    minmin = np.min(mindata)
    minmax = np.min(maxdata)
    maxmin = np.max(mindata)
    maxmax = np.max(maxdata)
    return (maxmin, minmax), (minmin, maxmax)

def fit_quantile_continuum(spec, quantile=90., wlrange=5., function="chebyshev", order=4, full_output=False):
    """ Calculate the 90% quantile (in a 5A range) and fit continuum to that """
    wave, flux, ivar = spec.dispersion, spec.flux, spec.ivar
    dwl = np.median(np.diff(wave))
    Npix = int(wlrange/dwl)
    if Npix % 2 == 0: Npix += 1
    quantflux = percentile_filter(flux, quantile, Npix)
    quantspec = specutils.Spectrum1D(wave, quantflux, ivar)
    # After filtering, do not iteratively reject points
    norm, cont, left, right = quantspec.fit_continuum(max_iterations=1, low_sigma_clip=90.0, high_sigma_clip=90.0,
                                                      order=order, function=function, full_output=True)
    newnorm = specutils.Spectrum1D(wave, flux/cont, ivar)
    if full_output:
        return newnorm, cont, left, right
    return newnorm

