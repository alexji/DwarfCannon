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
    for row in tab:
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
