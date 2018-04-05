import matplotlib.pyplot as plt
import numpy as np
import glob, os, sys, time, re
import read_data as rd; reload(rd)
import pandas as pd

import seaborn.apionly as sns

from astropy.io import ascii, fits
from astropy import table
from smh import specutils

import thecannon as tc

from fit_cannon import load_wave_flux_ivar, cut_pixels_1, cut_pixels_5param, cut_pixels_6param, cut_pixels_4000_6800

def plot(which_one=0):
    mtab = rd.load_roed_data()
    wave, flux, ivar = load_wave_flux_ivar()
    #wave, flux, ivar = cut_pixels_1(wave,flux,ivar)
    wave, flux, ivar = cut_pixels_4000_6800(wave,flux,ivar)
    
    # Construct some tags
    colordict = dict(zip(np.unique(mtab["Year"]), sns.hls_palette(len(np.unique(mtab["Year"])))))
    
    if which_one == 0:
        ## 2007/2008 
        title = "red = 2007/2008"
        figname = "one-one_20072008.png"
        colordict = {}
        for year in np.unique(mtab["Year"]):
            if year == "2008" or year == "2007": colordict[year] = "red"
            else: colordict[year] = "black"
        colors = map(lambda x: colordict[x], mtab["Year"])
    elif which_one == 1:
        ## Duplicate stars
        title = "duplicate spectra"
        figname = "one-one_duplicates.png"
        counts = pd.Series(mtab["Star"]).value_counts()
        N_repeat = np.sum(counts > 1)
        palette = sns.color_palette("Set1", N_repeat)
        ipalette = 0
        colordict = {}
        for star in np.unique(mtab["Star"]):
            if counts.loc[star] > 1:
                colordict[star] = palette[ipalette]
                ipalette += 1
            else: 
                colordict[star] = (0,0,0)
        colors = map(lambda x: colordict[x], mtab["Star"])
    elif which_one == 2:
        ## Code 1
        title = "code1"
        figname = "one-one_code1.png"
        code1tab = ascii.read("code1_data.txt")
        colors = np.array(["k" for i in range(len(mtab))])
        colors[code1tab["Index"]] = "r"
    elif which_one == 3:
        ## HB
        title = "HB"
        figname = "one-one_HB.png"
        colors = np.array(["k" for i in range(len(mtab))])
        colors[mtab["Cl"]=="HB"] = "r"
    else:
        raise ValueError("which_one is wrong: {}".format(which_one))
    
    # Load model
    #model = tc.CannonModel.read("initial_naive_train_ivar0.model")
    #test_labels, cov, metadata = model.test(flux,ivar)
    #np.save("initial_naive_train_ivar0_test.npy",test_labels)
    #test_labels = np.load("initial_naive_train_ivar0_test.npy")

    model = tc.CannonModel.read("initial_naive_train_ivar0_40006800.model")
    figname = "40006800_"+figname
    #test_labels, cov, metadata = model.test(flux,ivar)
    #np.save("initial_naive_train_ivar0_40006800_test.npy",test_labels)
    test_labels = np.load("initial_naive_train_ivar0_40006800_test.npy")

    assert np.all(wave == model.dispersion)
    label_names = model.vectorizer.label_names
    Nlabels = len(label_names)
    fig, axes = plt.subplots(Nlabels, figsize=(5, 4*Nlabels))
    for i,ax in enumerate(axes):
        true_labels = model.training_set_labels[:,i]
        pred_labels = test_labels[:,i]
        
        ax.scatter(true_labels, pred_labels, c=colors)
        all_x = np.concatenate((true_labels, pred_labels))
        xrange = (np.min(all_x), np.max(all_x))
        ax.set_xlim(xrange)
        ax.set_ylim(xrange)
        ax.plot(xrange, xrange, 'k:')
        ax.set_xlabel("True {}".format(label_names[i]))
        ax.set_ylabel("Pred {}".format(label_names[i]))
        diff = pred_labels - true_labels
        mu, med, sigma = np.mean(diff), np.median(diff), np.std(diff)
        ax.text(0.05, 0.90, r"$\mu = {0:.2f}$".format(mu),
            transform=ax.transAxes)
        ax.text(0.05, 0.84, r"med$ = {0:.2f}$".format(med),
            transform=ax.transAxes)
        ax.text(0.05, 0.78, r"$\sigma = {0:.2f}$".format(sigma),
            transform=ax.transAxes)
        
        #iibad = np.where(np.abs(diff/sigma) > 2)[0]
        #print "{} bad stars for {}".format(len(iibad), label_names[i], iibad)
        #print mtab["Star","Date","Teff","logg","[M/H]","[CH/Fe]","Vt","SN3950","SN4550"][iibad]
    axes[0].set_title(title)
    #fig.subplots_adjust(left=.18, right=.99, top=.97, bottom=.05, hspace=.22)
    fig.tight_layout()
    fig.savefig(figname, bbox_inches="tight")
    #plt.show()
    plt.close(fig)

if __name__=="__main__":
    plot(0)
    plot(1)
    plot(2)
    plot(3)
