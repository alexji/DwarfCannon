import numpy as np
import glob, os, sys, time, re
import read_data as rd; reload(rd)

from astropy.io import ascii, fits
from astropy import table
from smh import specutils

import thecannon as tc

def load_wave_flux_ivar():
    wave = rd.load_master_common_dispersion()
    flux, ivar = rd.load_stitched_flux_ivar()
    return wave, flux, ivar

def cut_pixels_1(wave, flux, ivar):
#    # Only keep pixels that are defined for all stars
#    good_pixel_mask = np.sum(np.isnan(flux),axis=0) == 0
#    # Further mask pixels that have unremoved cosmics etc
#    good_pixel_mask = np.logical_and(good_pixel_mask, np.all(np.abs(flux-1) < 1, axis=0))
#    print np.sum(good_pixel_mask)
#
#    # Mask the pixels
#    wave = wave[good_pixel_mask]
#    flux = flux[:, good_pixel_mask]
#    ivar = ivar[:, good_pixel_mask]
#    # Manually remove one outlier
#    # bad pixel: 3953.4970366153129
#    good_pixel_mask_2 = np.ones_like(wave, dtype=bool)
#    good_pixel_mask_2[1263] = False
#    wave = wave[good_pixel_mask_2]
#    flux = flux[:, good_pixel_mask_2]
#    ivar = ivar[:, good_pixel_mask_2]
#    
#    import matplotlib.pyplot as plt
#    fig, ax = plt.subplots(figsize=(20,5))
#    for i in range(len(mtab)):
#        ax.plot(wave, flux[i,:], ',', alpha=.2)
#    fig.savefig("training_pixels.png", bbox_inches='tight')
    
    ## Remove nan pixels
    ii = np.isnan(flux)
    flux[ii] = 1.0
    ivar[ii] = 0.0
    
    ## Remove outlier points
    ii = np.abs(flux - 1) > 1
    flux[ii] = 1.0
    ivar[ii] = 0.0
    
    # Cut on wavelength
    ii = ((wave > 3500) & (wave < 6800)) | ((wave > 8450) & (wave < 8700))
    wave = wave[ii]
    flux = flux[:,ii]
    ivar = ivar[:,ii]

    # Cut a few bad pixels manually. I found these by training
    # the cannon and finding abs(theta0-1) > 2
    # These are almost certainly data artifacts
    #In [35]: wave[np.where(np.abs(theta[:,0] - 1) > 2)[0]]
    #Out[35]: array([ 3864.13201439,  3953.45857401,  5068.62819978])
    bad_pixels = np.array([10193, 12532, 37860])
    mask = np.ones_like(wave, dtype=bool)
    mask[bad_pixels] = False
    wave = wave[mask]
    flux = flux[:,mask]
    ivar = ivar[:,mask]

    bad_pixels = np.array([2720, 37858])
    mask = np.ones_like(wave, dtype=bool)
    mask[bad_pixels] = False
    wave = wave[mask]
    flux = flux[:,mask]
    ivar = ivar[:,mask]
    
    return wave, flux, ivar

def cut_pixels_5param(wave, flux, ivar):
    bad_pixels = np.array([4944, 12548, 37856])
    mask = np.ones_like(wave, dtype=bool)
    mask[bad_pixels] = False
    wave = wave[mask]
    flux = flux[:,mask]
    ivar = ivar[:,mask]
    return wave, flux, ivar

def cut_pixels_6param(wave, flux, ivar):
    bad_pixels = np.array([4803, 12533])
    mask = np.ones_like(wave, dtype=bool)
    mask[bad_pixels] = False
    wave = wave[mask]
    flux = flux[:,mask]
    ivar = ivar[:,mask]

    bad_pixels = np.array([4819, 4943, 12533, 37854])
    mask = np.ones_like(wave, dtype=bool)
    mask[bad_pixels] = False
    wave = wave[mask]
    flux = flux[:,mask]
    ivar = ivar[:,mask]
    
    return wave, flux, ivar

if __name__=="__main__":
    mtab = rd.load_roed_data()
    training_labels = ["Teff", "logg", "[M/H]"]
    training_labels_2 = ["Teff", "logg", "[M/H]", "Vt", "[Ca I/Fe]"]
    training_labels_3 = ["Teff", "logg", "[M/H]", "Vt", "[Mg I/Fe]", "[Ca I/Fe]"]

    wave, flux, ivar = load_wave_flux_ivar()
    wave, flux, ivar = cut_pixels_1(wave, flux, ivar)
    
    ### Original training: 3 param, ivar0
    #print len(wave)
    #model = tc.CannonModel(
    #    mtab, flux, ivar,
    #    dispersion=wave,
    #    vectorizer=tc.vectorizer.PolynomialVectorizer(training_labels, 2))
    #theta, s2, metadata = model.train(threads=4)
    #model.write("initial_naive_train_ivar0.model", overwrite=True)
    #fig_theta = tc.plot.theta(model)
    #fig_theta.savefig("theta_ivar0.png",dpi=600)
    #
    #test_labels, cov, metadata = model.test(flux,ivar)
    #fig_comparison = tc.plot.one_to_one(model, test_labels)
    #fig_comparison.savefig("one-to-one_ivar0.png", dpi=300)

    ### 5param model, ivar0
    #wave, flux, ivar = cut_pixels_5param(wave, flux, ivar)
    #print len(wave)
    #model = tc.CannonModel(
    #    mtab, flux, ivar,
    #    dispersion=wave,
    #    vectorizer=tc.vectorizer.PolynomialVectorizer(training_labels_2, 2))
    #theta, s2, metadata = model.train(threads=4)
    #model.write("initial_naive_train_ivar0_5param.model", overwrite=True)
    #fig_theta = tc.plot.theta(model, indices=range(6))
    #fig_theta.savefig("theta_ivar0_5param.png",dpi=600)
    #test_labels, cov, metadata = model.test(flux,ivar)
    #fig_comparison = tc.plot.one_to_one(model, test_labels)
    #fig_comparison.savefig("one-to-one_ivar0_5param.png", dpi=300)

    ### 6param model, ivar0
    #wave, flux, ivar = cut_pixels_6param(wave, flux, ivar)
    #print len(wave)
    #model = tc.CannonModel(
    #    mtab, flux, ivar,
    #    dispersion=wave,
    #    vectorizer=tc.vectorizer.PolynomialVectorizer(training_labels_3, 2))
    #theta, s2, metadata = model.train(threads=4)
    #model.write("initial_naive_train_ivar0_6param.model", overwrite=True)
    #fig_theta = tc.plot.theta(model, indices=range(7))
    #fig_theta.savefig("theta_ivar0_6param.png",dpi=600)
    #test_labels, cov, metadata = model.test(flux,ivar)
    #fig_comparison = tc.plot.one_to_one(model, test_labels)
    #fig_comparison.savefig("one-to-one_ivar0_6param.png", dpi=300)


    ### 0th order train: this is not used anymore, it was with cutting inconsistent pixels
    #print len(wave)
    #model = tc.CannonModel(
    #    mtab, flux, ivar,
    #    dispersion=wave,
    #    vectorizer=tc.vectorizer.PolynomialVectorizer(training_labels, 2))
    #theta, s2, metadata = model.train(threads=2)
    #model.write("initial_naive_train.model", overwrite=True)
    #fig_theta = tc.plot.theta(model)
    #fig_theta.savefig("theta.png",dpi=600)
    
