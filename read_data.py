import numpy as np
from smh import specutils
#import matplotlib.pyplot as plt
from smh.specutils import Spectrum1D, motions
from astropy.io import ascii, fits
from astropy import table
import glob, os, sys, time, re
import pandas as pd
from collections import OrderedDict
from scipy import optimize, interpolate
from astropy.time import Time
from datetime import datetime
import cPickle as pickle

import utils

#
#manual_order_remap = {58:29, 59:30, 60:31, 61:57, 62:32, 63:33, 66:36, 67:37, 
#                      79:2, 80:64, 81:0, 82:0, 83:34, 84:29}
manual_order_remap = {79:2, 81:0, 82:0, 83:34}

def load_continuum_kwargs():
    """ Manually created continuum masks for big lines, just a temporary automatic thing """
    with open("all_kwargs_final.pkl","r") as fp:
        kwargs = pickle.load(fp)
    return kwargs

def make_order_labels():
    tab2, allspecs2 = load_master_table_and_spectra(rvcor_spectra=False)
    order_limits, all_order_labels = utils.match_orders(allspecs2, dwl=17)
    with open("order_labels.pkl","w") as fp:
        pickle.dump([order_limits, all_order_labels], fp)
def load_order_labels(manual_remap=True):
    with open("order_labels.pkl","r") as fp:
        order_limits, all_order_labels = pickle.load(fp)
    # Manual remapping of some order labels
    if manual_remap:
        manual_map = manual_order_remap
        for order_labels in all_order_labels:
            for a,b in manual_map.iteritems():
                while True:
                    try:
                        order_labels[order_labels.index(a)] = b
                    except ValueError:
                        break
    return order_limits, all_order_labels

def load_good_table_spectra_labels(rvcor_spectra=True, verbose=False):
    # Based on manual inspection, these are stars with bad orders (looks saturated)
    # The stars are all the ones observed in 2007 and 2008
    order_limits, all_order_labels = load_order_labels()
    bad_order_labels = [25,26,27,28,29,30,31,32,33,36,37,57,62,63,64,65,66,67]
    bad_order_stars = [3, 4, 5, 33, 35, 37, 50, 51, 52, 55, 61, 67, 73, 86, 89, 100, 111, 132, 133]
    tab, all_orders = load_master_table_and_spectra(rvcor_spectra=rvcor_spectra)
    num_orders_removed = 0
    for star_index in bad_order_stars:
        if verbose: print "Star {}".format(star_index)
        order_labels = all_order_labels[star_index]
        orders = all_orders[star_index]
        for bad_order_label in bad_order_labels:
            # Loop in case multiple orders have the same order label
            while True:
                try:
                    if verbose: print len(order_labels), order_labels
                    bad_index = order_labels.index(bad_order_label)
                except ValueError:
                    # this order number is not available anymore
                    break
                else:
                    # remove the order label
                    _ = order_labels.pop(bad_index)
                    # remove the order spectrum
                    _ = orders.pop(bad_index)
                    num_orders_removed += 1
    print "Removed {} bad orders".format(num_orders_removed)
    assert len(all_orders) == len(all_order_labels)
    for i in range(len(all_orders)):
        assert len(all_orders[i]) == len(all_order_labels[i])
    return tab, all_orders, all_order_labels

def load_master_table_and_spectra(rvcor_spectra=True):
    ## Load master table and associated files
    tab = load_master_table()

    ## Fill data from the actual files
    all_rv_spec = []
    vobss = []
    blue_min = []; blue_max = []
    red_min = []; red_max = []
    blue_norder = []; red_norder = []
    blue_min2 = []; blue_max2 = []
    red_min2 = []; red_max2 = []
    blue_norder2 = []; red_norder2 = []
    order_numss = []; order_numss2 = []
    for irow,row in enumerate(tab):
        #print row["blue_fname"], row["red_fname"]
        ordernumkey = None
        
        blue = read_multispec("data/{}".format(row["blue_fname"]))
        if blue[0].dispersion[0] > blue[-1].dispersion[0]:
            blue = blue[::-1]
        red  = read_multispec("data/{}".format(row["red_fname"]))
        if red[0].dispersion[0] > red[-1].dispersion[0]:
            red = red[::-1]
        vbar, vhel = motions.corrections_from_headers(blue[0].metadata)
        if row["vhel"]=="\\nodata":
            assert row["Star"]=="CS22940-077"
            vobs = +0.9
            vobss.append(+0.9) #manually did this in SMHr against HD140283
        else:
            vobs = float(row["vhel"]) - vhel.to('km/s').value
            vobss.append(vobs)
    
        # Before RV correction (good for instrument matching)
        blue_min.append(blue[0].dispersion[0])
        blue_max.append(blue[-1].dispersion[-1])
        blue_norder.append(len(blue))
        red_min.append(red[0].dispersion[0])
        red_max.append(red[-1].dispersion[-1])
        red_norder.append(len(red))
        
        
        
        orders = blue+red
        vcorr = -vobs
        order_nums = []; order_nums2 = []
        for order in orders:
            order.metadata["Star"] = row["Star"]
            order.metadata["StarIndex"] = row["Star"]+"-{:03}".format(irow)
            order.metadata["MasterIndex"] = irow
            if rvcor_spectra:
                order.redshift(vcorr)
            if ordernumkey is None:
                if int(order.metadata["beam"])==1:
                    ordernumkey = "aperture"
                else:
                    ordernumkey = "beam"
            order_nums.append(str(int(order.metadata[ordernumkey])))
    
        all_rv_spec.append(orders)
        order_nums = ",".join(order_nums)
        order_numss.append(order_nums)
        
        # After RV correction
        if rvcor_spectra:
            blue_min2.append(blue[0].dispersion[0])
            blue_max2.append(blue[-1].dispersion[-1])
            blue_norder2.append(len(blue))
            red_min2.append(red[0].dispersion[0])
            red_max2.append(red[-1].dispersion[-1])
            red_norder2.append(len(red))
        
    tab.add_column(tab.Column(vobss, "vobs"))
    tab.add_column(tab.Column(blue_min, "blue_wlmin"))
    tab.add_column(tab.Column(blue_max, "blue_wlmax"))
    tab.add_column(tab.Column(red_min, "red_wlmin"))
    tab.add_column(tab.Column(red_max, "red_wlmax"))
    tab.add_column(tab.Column(blue_norder, "blue_norder"))
    tab.add_column(tab.Column(red_norder, "red_norder"))
    if rvcor_spectra:
        tab.add_column(tab.Column(blue_min2, "blue_wlmin2"))
        tab.add_column(tab.Column(blue_max2, "blue_wlmax2"))
        tab.add_column(tab.Column(red_min2, "red_wlmin2"))
        tab.add_column(tab.Column(red_max2, "red_wlmax2"))
        tab.add_column(tab.Column(blue_norder2, "blue_norder2"))
        tab.add_column(tab.Column(red_norder2, "red_norder2"))
    tab.add_column(tab.Column(order_numss, "order_nums"))    
    
    return tab, all_rv_spec
    
def sort_specs_by_order_label(allspecs=None, all_order_labels=None, rvcor_spectra=True):
    if allspecs is None:
        _, allspecs = load_master_table_and_spectra(rvcor_spectra=rvcor_spectra)
    if all_order_labels is None:
        _, all_order_labels = load_order_labels()
    ## Verify that everything is the same size
    assert len(allspecs) == len(all_order_labels), (len(allspecs), len(all_order_labels))
    for allspec, order_labels in zip(allspecs, all_order_labels):
        assert len(allspec) == len(order_labels), (len(allspec), len(order_labels))
    ## Make a dictionary
    unique_labels = np.unique(utils.flatten_list_of_lists(all_order_labels))
    sorted_specs = dict(zip(unique_labels, [[] for _ in unique_labels]))
    for allspec, order_labels in zip(allspecs, all_order_labels):
        for spec, order_label in zip(allspec, order_labels):
            sorted_specs[order_label] += [spec]
    return sorted_specs

def load_master_table():
    def month2month(monthstr):
        data = {"Jan":1,"Feb":2,"Mar":3,"Apr":4,"May":5,"Jun":6,
                "Jul":7,"Aug":8,"Sep":9,"Oct":10,"Nov":11,"Dec":12}
        return "{:02}".format(data[monthstr])

    ## Load raw table
    tab = ascii.read("tab02.tex", delimiter='&',guess=False,comment="%")
    instruments = map(lambda x: x.split("/")[1], tab["Telescope/Instrument"])
    years = map(lambda x: x.split()[0], tab["Date"])
    months = map(lambda x: month2month(x.split()[1]), tab["Date"])
    days = map(lambda x: x.split()[2], tab["Date"])
    tab.add_column(table.Column(instruments, name="Instrument"))
    tab.add_column(table.Column(years, name="Year"))
    tab.add_column(table.Column(months, name="Month"))
    tab.add_column(table.Column(days, name="Day"))
    
    ## Match table to fnames
    fnames = map(os.path.basename, glob.glob("data/*.fits"))
    def calc_mjd(fname):
        with fits.open(fname) as hdulist:
            header = hdulist[0].header
            ut_date = header["UT-DATE"]
            ut_time = header["UT-TIME"] # time in seconds
        for sep in ["-","/"]:
            try:
                year, month, day = ut_date.split(sep)
            except:
                pass
            else:
                year = int(year); month = int(month); day = int(day)
                break
        hour = int(ut_time/3600.)
        minute = int((ut_time - 3600*hour)/60.)
        second = int((ut_time - 3600*hour - 60*minute)/60.)
        try:
            t = Time(datetime(year, month, day, hour, minute, second), scale='utc')
        except:
            print ut_date, ut_time
            print year, month, day
            print hour, minute, second
            raise
        return t.jd
    def get_goodname(name):
        if name.startswith("bdm"):
            name = "_".join(name.split("-"))
            name = "bd-"+name[3:]
        if name.startswith("bdp"):
            name = "_".join(name.split("-"))
            name = "bd+"+name[3:]
        if name.startswith("cdm"):
            name = "_".join(name.split("-"))
            name = "cd-"+name[3:]
        if name.startswith("cdp"):
            name = "_".join(name.split("-"))
            name = "cd+"+name[3:]
        if "blue" in name:
            return name.split("blue")[0].upper()
        if "red" in name:
            return name.split("red")[0].upper()
    def find_row(fname, tab):
        fname = os.path.basename(fname)
        goodname = get_goodname(fname)
        name, other = fname.split("_multi_")
        assert other.endswith(".fits")
        year = other[0:4]
        month = other[4:6]
        observer = other[6:9]
        if other[9]=="-":
            number = other[10]
        else:
            number = "1"
        ii = np.array(goodname == tab["Star"]) & np.array(year == tab["Year"]) & np.array(month == tab["Month"])
        assert np.sum(ii) > 0, fname
        if np.sum(ii) != 1:
            mjd = calc_mjd("data/"+fname)
            subtab = tab[ii]
            ix = np.argmin(np.abs(subtab["MJD"] - mjd))
            day = subtab[ix]["Day"]
            ii = np.array(goodname == tab["Star"]) & np.array(year == tab["Year"]) & np.array(month == tab["Month"]) \
                  & np.array(day == tab["Day"])
            assert np.sum(ii) == 1, fname
            ix = np.where(ii)[0][0]
            #print "MULTIPLE:",mjd, fname,np.sum(ii), ix, list(tab[ii]["MJD"]) #np.min(np.abs(tab["MJD"]-mjd))
        return np.where(ii)[0][0]
    all_fnames_blue = []
    all_out_blue = []
    all_fnames_red = []
    all_out_red = []
    for fname in fnames:
        out = find_row(fname, tab)
        if "blue" in fname:
            all_fnames_blue.append(fname)
            all_out_blue.append(out)
        else:
            all_fnames_red.append(fname)
            all_out_red.append(out)
    all_fnames_blue = np.array(all_fnames_blue)
    all_fnames_red = np.array(all_fnames_red)
    assert np.all(np.array(all_out_blue) == np.array(all_out_red))
    assert len(all_out_blue) == len(np.unique(all_out_blue)) == len(fnames)/2 == len(tab)
    ii = np.argsort(all_out_blue)
    tab.add_column(tab.Column(all_fnames_blue[ii],"blue_fname"))
    tab.add_column(tab.Column(all_fnames_red[ii], "red_fname"))
    
    return tab



def read_multispec(fname, full_output=False):
    """
    There are some files that are not reduced with Dan Kelson's current pipeline version.
    So we have to read those manually and define ivar
    """
    # Hardcoded file with old CarPy format: 5 bands instead of 7
    if "hd13979red_multi_200311ibt" in fname:
        WAT_LENGTH=67
    else:
        WAT_LENGTH=68
    
    with fits.open(fname) as hdulist:
        assert len(hdulist)==1, len(hdulist)
        header = hdulist[0].header
        data = hdulist[0].data
        # orders x pixels
        # assert len(data.shape)==2, data.shape

        metadata = OrderedDict()
        for k, v in header.items():
            if k in metadata:
                metadata[k] += v
            else:
                metadata[k] = v

    ## Compute dispersion
    assert metadata["CTYPE1"].upper().startswith("MULTISPE") \
        or metadata["WAT0_001"].lower() == "system=multispec"
    # Join the WAT keywords for dispersion mapping.
    i, concatenated_wat, key_fmt = (1, str(""), "WAT2_{0:03d}")
    while key_fmt.format(i) in metadata:
        value = metadata[key_fmt.format(i)]
        concatenated_wat += value + (" "  * (WAT_LENGTH - len(value)))
        i += 1
    # Split the concatenated header into individual orders.
    order_mapping = np.array([map(float, each.rstrip('" ').split()) \
        for each in re.split('spec[0-9]+ ?= ?"', concatenated_wat)[1:]])
    dispersion = np.array(
        [specutils.spectrum.compute_dispersion(*mapping) for 
         mapping in order_mapping])
    
    if len(data.shape)==2:
        ## Compute flux
        flux = data
        flux[0 > flux] = np.nan

        ## Compute ivar assuming Poisson noise
        ivar = 1./flux
        ivar[0 > flux] = 0
        code=1
    elif len(data.shape)==3:
        flux = data[1]
        ivar = data[2]**(-2)
        flux[0 > flux] = np.nan
        ivar[0 > flux] = 0
        code=2

    ## Turn into orders
    orders = [Spectrum1D(dispersion=d, flux=f, ivar=i, metadata=metadata.copy()) \
              for d,f,i in zip(dispersion, flux, ivar)]
    for order, mapping in zip(orders, order_mapping):
        order.metadata["aperture"] = mapping[0]
        order.metadata["beam"] = mapping[1]
    if full_output: return orders, code
    return orders


def load_order_table():
    return table.Table(np.load("order_dispersion_table.npy"))
def load_tabulated_dispersion(minmax, order):
    assert minmax in ["min","max"], minmax
    tab = load_order_table()
    ix = np.where(order == tab["order"])[0][0]
    col1 = "{}wl1".format(minmax)
    col2 = "{}wl2".format(minmax)
    return np.arange(tab[ix][col1], tab[ix][col2] + tab[ix]["dwl"], tab[ix]["dwl"])
    
def load_min_dispersion(order):
    data = np.load("order_dispersion_table.npy")
    ix = np.where(order == data[:,0])[0][0]
    return np.arange(data[ix,1], data[ix,2]+data[ix,5], data[ix,5])
def load_max_dispersion(order):
    data = np.load("order_dispersion_table.npy")
    ix = np.where(order == data[:,0])[0][0]
    return np.arange(data[ix,3], data[ix,4]+data[ix,5], data[ix,5])

def load_flux_order(minmax,order):
    assert minmax in ["min","max"], minmax
    return np.load("data_common_dispersion/{}flux_order{:02}.npy".format(minmax,order))
def load_ivar_order(minmax,order):
    assert minmax in ["min","max"], minmax
    return np.load("data_common_dispersion/{}ivar_order{:02}.npy".format(minmax,order))
def load_normflux_order(minmax,order):
    assert minmax in ["min","max"], minmax
    return np.load("data_common_dispersion/{}normflux_order{:02}.npy".format(minmax,order))
def load_normivar_order(minmax,order):
    assert minmax in ["min","max"], minmax
    return np.load("data_common_dispersion/{}normivar_order{:02}.npy".format(minmax,order))

def load_interp_spec_order(minmax, index, order, fluxdata=None, ivardata=None):
    assert minmax in ["min","max"], minmax
    if fluxdata is None: fluxdata = load_flux_order(minmax, order)
    if ivardata is None: ivardata = load_ivar_order(minmax, order)
    wave = load_tabulated_dispersion(minmax, order)
    return Spectrum1D(wave, fluxdata[index,:], ivardata[index,:])
def load_norm0_spec_order(minmax, index, order, fluxdata=None, ivardata=None):
    assert minmax in ["min","max"], minmax
    if fluxdata is None: fluxdata = load_normflux_order(minmax, order)
    if ivardata is None: ivardata = load_normivar_order(minmax, order)
    wave = load_tabulated_dispersion(minmax, order)
    return Spectrum1D(wave, fluxdata[index,:], ivardata[index,:])

#def load_min_star_order(star, order):
#    fname = "data_common_dispersion/{}_{}min.fits".format(star, order)
#    return Spectrum1D.read(fname)
#def load_max_star_order(star, order):
#    fname = "data_common_dispersion/{}_{}max.fits".format(star, order)
#    return Spectrum1D.read(fname)
#def load_min_allstars_order(order):
#    fnames = glob.glob("data_common_dispersion/*_{}min.fits".format(order))
#    return [Spectrum1D.read(fname) for fname in fnames]
#def load_max_allstars_order(order):
#    fnames = glob.glob("data_common_dispersion/*_{}max.fits".format(order))
#    return [Spectrum1D.read(fname) for fname in fnames]
