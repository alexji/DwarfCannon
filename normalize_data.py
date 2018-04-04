"""
The steps are:
(1) Preliminary sort of orders (rd.make_order_labels, utils.match_orders)
(2) Manually remap some orders (rd.manual_order_remap, rd.load_order_labels)
    plot_stuff_for_order_remapping()    
(3) Drop some data that is bad due to saturation (rd.load_good_table_spectra_labels)
    For observations in 2007/2008, the LL CCD failed and has since been replaced by an E2V CCD (I. Thompson)
    I manually picked out the stars and orders that had this problem.
(4) Define wavelength ranges for each order: a minimum and a maximum range (utils.find_minrange_maxrange)
    And also a dwl for the order.
    Linearly interpolate onto this new (linear) wavelength range for each order.
    TODO
(5) Fit preliminary continuum:
    
    

One of the key steps here is to define the orders.
For the initial reduction (normalization) step, each order needs to be treated independently.

There are challenges:
(1) large velocity range
(2) inhomogeneous data (different orders for a particular spectrum)

"""

import numpy as np
from smh import specutils
import matplotlib.pyplot as plt
from smh.specutils import Spectrum1D, motions
from smh.specutils.spectrum import common_dispersion_map2 as common_dispersion_map
from astropy.io import ascii, fits
from astropy import table
import glob, os, sys, time, re
import pandas as pd
from collections import OrderedDict
from scipy import optimize, interpolate
from astropy.time import Time
import cPickle as pickle


import read_data as rd
import utils


def plot_stuff_for_order_remapping():
    """ Make a plot showing order limits and how I remapped it """
    order_limits, all_order_labels = rd.load_order_labels(manual_remap=True)
    order_labels_flat = []
    for order_labels in all_order_labels:
        order_labels_flat += order_labels
    order_labels_flat = np.array(order_labels_flat)

    fig, ax = plt.subplots(figsize=(15,9))
    for i,limit in enumerate(order_limits):
        l, = ax.plot([i,i],limit)
        color = l.get_color()
        if i >= 57:
            ax.hlines(limit,-5,85,color=color)
        ax.text(i,limit[0]-10,str(np.sum(order_labels_flat == i)), ha='center', va='top', fontsize=8)
    ax.set_xlim(-5,85)
    fig.savefig("order_remapping.pdf")
    plt.show()

def interpolate_onto_common_dispersion():
    """
    Define common dispersion for each order and interpolate onto that dispersion
    For each order, create a separate file
    """
    tab, allspecs, all_order_labels = rd.load_good_table_spectra_labels(rvcor_spectra=True)
    Nspec = len(tab)
    sorted_specs = rd.sort_specs_by_order_label(allspecs, all_order_labels)
    order_labels = np.sort(sorted_specs.keys())
    data = []
    for order in order_labels:
        specs = sorted_specs[order]
        waves = [spec.dispersion for spec in specs]
        (minwl1, minwl2), (maxwl1, maxwl2) = utils.find_minrange_maxrange(waves)
        
        disp = common_dispersion_map(specs)
        dwl = np.median(np.diff(disp))

        minwldisp = np.arange(minwl1, minwl2+dwl, dwl)
        maxwldisp = np.arange(maxwl1, maxwl2+dwl, dwl)
        print "{:02} {:7.2f}-{:7.2f}, {:.3f} (Npt={}, Nspec={})".format(order, minwl1, minwl2, dwl, len(minwldisp), len(specs))
        print "{:02} {:7.2f}-{:7.2f}, {:.3f} (Npt={}, Nspec={})".format(order, maxwl1, maxwl2, dwl, len(maxwldisp), len(specs))
        data.append([order, minwl1, minwl2, maxwl1, maxwl2, dwl])
        
        fluxdatmin = np.full((Nspec, len(minwldisp)), np.nan)
        ivardatmin = np.full((Nspec, len(minwldisp)), np.nan)
        fluxdatmax = np.full((Nspec, len(maxwldisp)), np.nan)
        ivardatmax = np.full((Nspec, len(maxwldisp)), np.nan)
        for spec in specs:
            starindex = spec.metadata["MasterIndex"]
            #starname = spec.metadata["StarIndex"]
            #minoutfname = "data_common_dispersion/{}_{}min.fits".format(starname, order)
            #maxoutfname = "data_common_dispersion/{}_{}max.fits".format(starname, order)
            specmin = spec.linterpolate(minwldisp, fill_value=np.nan)
            fluxdatmin[starindex,:] = specmin.flux
            ivardatmin[starindex,:] = specmin.ivar
            #specmin.write(minoutfname)
            specmax = spec.linterpolate(maxwldisp, fill_value=np.nan)
            fluxdatmax[starindex,:] = specmax.flux
            ivardatmax[starindex,:] = specmax.ivar
            #specmax.write(maxoutfname)
        np.save("data_common_dispersion/minflux_order{:02}".format(order),fluxdatmin)
        np.save("data_common_dispersion/minivar_order{:02}".format(order),ivardatmin)
        np.save("data_common_dispersion/maxflux_order{:02}".format(order),fluxdatmax)
        np.save("data_common_dispersion/maxivar_order{:02}".format(order),ivardatmax)
            
    tab = table.Table(rows=data, names=["order","minwl1","minwl2","maxwl1","maxwl2","dwl"])
    tab.write("order_dispersion_table.dat",format="ascii.fixed_width_two_line")
    np.save("order_dispersion_table.npy",tab.as_array())

def make_norm0_spectra():
    order_table = table.Table(np.load("order_dispersion_table.npy"))
    N = len(order_table)
    assert N == 81
    regex = re.compile("data_common_dispersion/(.*)\_(\d+)(...)\.fits")
    fnames = glob.glob("data_common_dispersion/*")
    for fname in fnames:
        if fname.startswith("norm0"): continue
        print fname
        star, order, minmax = regex.findall(fname)[0]
        spec = Spectrum1D.read(fname)
        norm = utils.fit_quantile_continuum(spec, order=4, Niter=5)
        norm.write("data_common_dispersion/norm0_{}_{}{}.fits".format(star, order, minmax))

if __name__=="__main__":
    interpolate_onto_common_dispersion()
    make_norm0_spectra()
    #for i, order_num in enumerate(np.sort(order_table["order"])):
    #    minstarorders = rd.load_min_allstars_order(order_num)
    #    maxstarorders = rd.load_min_allstars_order(order_num)
        
def tmp():
    """ Plot all spectra on common dispersion map, norm and not"""
    order_table = table.Table(np.load("order_dispersion_table.npy"))
    N = len(order_table)
    assert N == 81
    # 
    fig1, axes1 = plt.subplots(9,9,figsize=(9*8,9*4))
    fig2, axes2 = plt.subplots(9,9,figsize=(9*8,9*4))
    fig3, axes3 = plt.subplots(9,9,figsize=(9*8,9*4))
    fig4, axes4 = plt.subplots(9,9,figsize=(9*8,9*4))
    for i, order_num in enumerate(np.sort(order_table["order"])):
        ax1 = axes1.flat[i]; ax2 = axes2.flat[i]
        ax3 = axes3.flat[i]; ax4 = axes4.flat[i]
        minstarorders = rd.load_min_allstars_order(order_num)
        for spec in minstarorders:
            ax1.plot(spec.dispersion, spec.flux, lw=.5, alpha=.1)
            norm = utils.fit_quantile_continuum
        maxstarorders = rd.load_min_allstars_order(order_num)
        
    

def plot_norm_specs_raw():
#if __name__=="__main__":
    cont_kwargs = rd.load_continuum_kwargs()
    order_limits, all_order_labels = rd.load_order_labels()
    order_labels_flat = utils.flatten_list_of_lists(all_order_labels)
    
    ### Use these for all orders including bad ones
    #tab, allspecs = rd.load_master_table_and_spectra(rvcor_spectra=True)
    #suffix = "orig"
    
    ### Use these for removing the bad orders
    tab, allspecs, all_order_labels = rd.load_good_table_spectra_labels(rvcor_spectra=True)
    suffix = "cut0708"
    
    xlims = dict(zip(np.arange(len(order_limits)), [[np.inf, -np.inf] for x in range(len(order_limits))]))

    # Plot stacked normalized rv spectra
    fig, axes = plt.subplots(17,5,figsize=(5*8,17*4))
    fig2list = []
    axes2list = []
    for row, allspec, order_labels in zip(tab, allspecs, all_order_labels):
        fig2, axes2 = plt.subplots(17,5,figsize=(5*8,17*4))
        fig2list.append(fig2)
        axes2list.append(axes2)
        for spec, order_label in zip(allspec, order_labels):
            if order_label in cont_kwargs:
                ax = axes.flat[order_label]
                kwargs = cont_kwargs[order_label]
                norm = spec.fit_continuum(**kwargs)
                l, = ax.plot(norm.dispersion, norm.flux, lw=.5, alpha=.1)
                xlims[order_label][0] = min(xlims[order_label][0], spec.dispersion[0])
                xlims[order_label][1] = max(xlims[order_label][1], spec.dispersion[-1])
                
                ax2 = axes2.flat[order_label]
                ax2.plot(norm.dispersion, norm.flux, lw=1, alpha=1, color=l.get_color())
            else:
                print "missing {}".format(order_label)
    for j,ax in enumerate(axes.flat):
        ax.set_title("{:2} ({})".format(j, np.sum(j == order_labels_flat)))
        ax.set_ylim(0,1.2)
        try:
            ax.set_xlim(utils.round10down(xlims[j][0]), utils.round10up(xlims[j][1]))
        except:
            pass
    for i,(fig2,axes2) in enumerate(zip(fig2list,axes2list)):
        for j, ax in enumerate(axes2.flat):
            ax.set_title("{:2} ({})".format(j, np.sum(j == order_labels_flat)))
            ax.set_ylim(0,1.2)
            try:
                ax.set_xlim(utils.round10down(xlims[j][0]), utils.round10up(xlims[j][1]))
            except:
                pass
        fig2.savefig("normalized_order_figures/star{:03}_norm_{}.pdf".format(i, suffix), bbox_inches="tight")
        plt.close(fig2)
        
    fig.savefig("all_norm_orders_1_{}.png".format(suffix))
    plt.close(fig)
    
#if __name__=="__main__":
def tmp():
#    tab, allspecs = rd.load_master_table_and_spectra(rvcor_spectra=True)
#    order_limits, all_order_labels = rd.load_order_labels()
    
    
    #tab, allspecs, all_order_labels = rd.load_good_table_spectra_labels(rvcor_spectra=True)
    #flatspecs = []
    #for orders in allspecs:
    #    flatspecs += orders
    #dispersion = common_dispersion_map(flatspecs)
    #np.save("super_common_dispersion.npy",dispersion)
    
    #from scipy import signal
    #dispersion = np.load("super_common_dispersion.npy")
    #min_dwl = .03
    #ii = np.diff(dispersion) > min_dwl
    #plt.plot(dispersion[:-1],np.diff(dispersion))
    ##plt.plot(dispersion[ii][:-1],np.diff(dispersion[ii]))
    ##plt.plot(dispersion, 
    #plt.gcf().savefig("dispersion_diff.pdf")
    #plt.show()
    
    tab, allspecs, all_order_labels = rd.load_good_table_spectra_labels(rvcor_spectra=True)
    sorted_specs = rd.sort_specs_by_order_label(allspecs, all_order_labels)
    order_dispersions = {}
    fig, ax = plt.subplots(figsize=(20,10))
    for order, specs in sorted_specs.iteritems():
        disp = common_dispersion_map(specs)
        wl1, wl2 = disp[0], disp[-1]
        order_dispersions[order] = disp
        l, = ax.plot(disp[:-1], np.diff(disp))
        median_disp = np.median(disp)
        median_dx = np.median(np.diff(disp))
        newdisp = np.arange(wl1,wl2+median_dx,median_dx)
        ax.plot([wl1,wl2], [median_dx,median_dx], color=l.get_color(), alpha=.5, lw=10)
        
        ax.text(median_disp, median_dx, "{:02} {:.3f} (N={})".format(order, median_dx, len(newdisp)), color=l.get_color(),
                ha='center', va='bottom', fontsize=8)
    fig.savefig("order_dispersion_diff.png", bbox_inches='tight')
    plt.show()


if __name__=="__main__":
    #plot_stuff_for_order_remapping()
    pass
