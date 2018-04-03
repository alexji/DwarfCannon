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

def find_outlier_orders():
    cont_kwargs = rd.load_continuum_kwargs()
    tab, allspecs = rd.load_master_table_and_spectra(rvcor_spectra=True)
    order_limits, all_order_labels = rd.load_order_labels()
    
    pass

def plot_norm_specs_raw():
    cont_kwargs = rd.load_continuum_kwargs()
    tab, allspecs = rd.load_master_table_and_spectra(rvcor_spectra=True)
    order_limits, all_order_labels = rd.load_order_labels()
    order_labels_flat = utils.flatten_list_of_lists(all_order_labels)

    xlims = dict(zip(np.arange(len(order_limits)), [[np.inf, -np.inf] for x in range(len(order_limits))]))

    # Plot stacked normalized rv spectra
    fig, axes = plt.subplots(17,5,figsize=(5*8,17*4))
    fig2list = []
    axes2list = []
    for row, allspec, order_labels in zip(tab, allspecs, all_order_labels):
        order_nums = map(int, row["order_nums"].split(","))
        assert len(order_nums) == len(allspec)
        fig2, axes2 = plt.subplots(17,5,figsize=(5*8,17*4))
        fig2list.append(fig2)
        axes2list.append(axes2)
        for order_num, spec, order_label in zip(order_nums, allspec, order_labels):
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
        fig2.savefig("normalized_order_figures/star{:03}_norm.pdf".format(i), bbox_inches="tight")
        plt.close(fig2)
            
    fig.savefig("test.png")
    plt.close(fig)
    

def plot_outliers():
    """ An attempt to automatically identify outlier orders (does not work great) """
    sorted_specs = rd.sort_specs_by_order_label()
    outlier_dict = {}
    cont_kwargs = rd.load_continuum_kwargs()
    outlier_const = 2
    for order_label, speclist in sorted_specs.iteritems():
        fig, ax = plt.subplots(figsize=(12,6))
        N = len(speclist)
        if N == 0: continue
        common_dispersion = common_dispersion_map(speclist)
        fluxes = np.zeros((N, len(common_dispersion)))
        for i in range(N):
            normspec = speclist[i].fit_continuum(**cont_kwargs[order_label])
            fluxes[i,:] = normspec.linterpolate(common_dispersion, fill_value = np.nan).flux
        q1 = np.nanpercentile(fluxes, 25, axis=0)
        q2 = np.nanpercentile(fluxes, 50, axis=0)
        q3 = np.nanpercentile(fluxes, 75, axis=0)
        IQR = q3 - q1
        upper_outlier_bound = q3 + outlier_const*IQR
        lower_outlier_bound = q1 - outlier_const*IQR
        upper_outlier = fluxes > upper_outlier_bound
        lower_outlier = fluxes < lower_outlier_bound
        outlier_pixels = upper_outlier | lower_outlier
        num_outlier_pixels = np.nansum(outlier_pixels, axis=1)

        print num_outlier_pixels
        for i in range(N):
            if num_outlier_pixels[i] > 200:
                color = 'r'
                lw = 1
                alpha = .1
            else:
                color = 'k'
                lw = .5
                alpha = .1
            normspec = speclist[i].fit_continuum(**cont_kwargs[order_label])
            ax.plot(normspec.dispersion, normspec.flux, lw=lw, alpha=alpha, color=color)
        ax.plot(common_dispersion, upper_outlier_bound, 'b:')
        ax.plot(common_dispersion, lower_outlier_bound, 'b:')
        #ax.set_xlim(utils.round10down(common_dispersion[0]), utils.round10up(common_dispersion[-1]))
        ax.set_xlim(common_dispersion[0], common_dispersion[-1])
        ax.set_ylim(0,1.5)
        #plt.figure()
        #plt.hist(num_outlier_pixels)
        #plt.show()
        fig.savefig("normalized_order_figures/outliers_{:02}.pdf".format(order_label), bbox_inches="tight")
        plt.close(fig)

if __name__=="__main__":
#def tmp():
    tab, allspecs = rd.load_master_table_and_spectra(rvcor_spectra=True)
    order_limits, all_order_labels = rd.load_order_labels()
    
    
