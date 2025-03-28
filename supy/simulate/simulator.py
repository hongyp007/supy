#	22.12.15 created by Gregory S.H. Paek
#	23.02.20 modified by Donggeun Tak

import warnings
warnings.filterwarnings('ignore')
import glob
import numpy as np
from astropy.table import Table, vstack, QTable, join
from astropy.io import ascii

from astropy import units as u
from astropy import constants as const

from scipy.optimize import curve_fit

from scipy.special import erf
from scipy.integrate import trapz
from scipy.stats import norm

import speclite.filters

#	Plot presetting
import matplotlib.pyplot as plt
import matplotlib as mpl

from pathlib import Path
from . import utils
from .filter import Filter
from . import response as res

SCRIPT_DIR = str(Path(__file__).parent.absolute())

#
mpl.rcParams["axes.titlesize"] = 14
mpl.rcParams["axes.labelsize"] = 20
plt.rcParams['savefig.dpi'] = 500
plt.rc('font', family='serif')
#

#
def synth_phot(wave, flux, wave_lvf, resp_lvf, tol=1e-3, return_photonrate = False):
    """
    Quick synthetic photometry routine.

    Parameters
    ----------
    wave : `numpy.ndarray`
        wavelength of input spectrum.
    flux : `numpy.ndarray`
        flux density of input spectrum in f_nu unit
        if `return_countrate` = True, erg/s/cm2/Hz is assumed
    wave_lvf : `numpy.ndarray`
        wavelength of the response function
    resp_lvf : `numpy.ndarray`
        response function. assume that this is a QE.
    tol : float, optional
        Consider only wavelength range above this tolerence (peak * tol).
        The default is 1e-3.

    Returns
    -------
    synthethic flux density in the input unit
        if return_photonrate = True, photon rates [ph/s/cm2]

    """
    index_filt, = np.where(resp_lvf > resp_lvf.max()*tol)

    index_flux, = np.where(np.logical_and( wave > wave_lvf[index_filt].min(), 
                                           wave < wave_lvf[index_filt].max() ))

    wave_resamp = np.concatenate( (wave[index_flux], wave_lvf[index_filt]) )
    wave_resamp.sort()
    wave_resamp = np.unique(wave_resamp)
    flux_resamp = np.interp(wave_resamp, wave, flux)
    resp_resamp = np.interp(wave_resamp, wave_lvf, resp_lvf)

    if return_photonrate:
        h_planck = 6.626e-27 # erg/Hz
        return trapz(resp_resamp / wave_resamp * flux_resamp, wave_resamp) / h_planck
        
    return trapz(resp_resamp / wave_resamp * flux_resamp, wave_resamp) \
         / trapz(resp_resamp / wave_resamp, wave_resamp)


def calculate_aperture_fraction(seeing, optfactor, figure=True):
	# np.random.seed(0)
	# seeing = 1.5
	# optfactor = 0.6731
	mu = 0.0
	sigma = seeing*2.3548

	x = np.linspace(-8, 8, 1000)
	y = (1 / np.sqrt(2 * np.pi * sigma**2)) * np.exp(-(x-mu)**2 / (2 * sigma**2))
	y_cum = 0.5 * (1 + erf((x - mu)/(np.sqrt(2 * sigma**2))))

	indx_aperture = np.where(
		(x>-sigma*optfactor) &
		(x<+sigma*optfactor)
	)
	xaper = x[indx_aperture]
	yaper = y[indx_aperture]

	frac = np.sum(yaper)/np.sum(y) 

	if figure:
		plt.plot(x, y, alpha=0.7, label=f'PDF of N(0, {sigma:1.3f})')
		plt.plot(xaper, yaper, alpha=0.75, label=f'Aperture ({frac*1e2:.1f}%)', lw=5,)
		plt.xlabel('x', fontsize=20)
		plt.ylabel('f(x)', fontsize=20)
		plt.legend(loc='lower center', fontsize=14)
		plt.show()
	return frac



#----------------------------------------------------------------
#	Main class
#----------------------------------------------------------------
class SevenDT(Filter):
	#	Optics info.

	_optics_info = {
	    "d": 50.5,                   # effective diameter [cm]
	    "d_obscuration": 29.8,       # Central Obscuration (diameter)
	    "d_efl": 1537.3,             # [mm]
	    "array": 'CMOS',             # detector array type
	    "dQ_RN": 3.0,                # [e], readout noise 
	    "I_dark": 0.01,              # [e/s], dark current
	    "pixel_size": 3.76,          # [um], "pitch"
	    "theta_pixel": 0.517,        # [arcsec], pixel scale 
	    "nxpix": 9576,               # [pixels], detector format, approx. 9k x 6k
	    "nypix": 6388,               # [pixels], detector format, approx. 9k x 6k
	}
	_optics_info["d_eff"] = np.sqrt(_optics_info["d"]**2-_optics_info["d_obscuration"]**2)
	
	_refdata = Path(f"{SCRIPT_DIR}/refdata")
	
	def __init__(self, filterset=None, verbose=True):
		self.verbose = verbose
		super().__init__(filterset=filterset, verbose=verbose)

		
	@property
	def optics_info(self):
		return self._optics_info

	@property
	def aperture_info(self):
		return self._aperture_info
	
	@property
	def echo_optics(self):
		print(f'D             : {self.optics_info["d"]}cm')
		print(f'D_obscuration : {self.optics_info["d_obscuration"]}cm')
		print(f'Deff          : {self.optics_info["d_eff"]:1.3f}cm')
	
	@property
	def echo_aperture(self):
		d = self._aperture_info["aperture_radius_arcsec"]*2
		print("Aperture radius   : {:1.3f} arcsec".format(self.aperture_info["aperture_radius_arcsec"]))
		print("Aperture radius   : {:1.3f} pix".format(self.aperture_info["aperture_radius_pix"]))
		print("fwhm_seeing       : {:1.3f} arcsec".format(self.aperture_info["seeing"]))
		print("exposure          : {:g} second".format(self.exposure))
		print("Aperture Diameter : {:1.3f} arcsec".format(d))
		print("Aperture Diameter : {:1.3f} pixel".format(d/self.optics_info["theta_pixel"]))
		print("SEEING*N Diameter : {:1.3f}".format(d/self.aperture_info["seeing"]))
		print("Aperture Area     : {:1.3f} arcsec2".format(np.pi*(d/2)**2))
		print("Aperture Area     : {:1.3f} pixel2".format(self.aperture_info["n_aperture_pix"]))

	def initalize(self, exposure, fwhm_seeing, **kwargs):
		totrsptbl = self._calculate_response()
		self._get_phot_aperture(exposure=exposure, fwhm_seeing=fwhm_seeing, optfactor=kwargs.pop("optfactor", 1.0), verbose=False)
		self._get_depth_table(Nsigma=kwargs.pop("Nsigma", 5))
		self._get_speclite()

	def _calculate_response(self, QE="default", group='sdt_default', **kwargs):
		filterColors = self._color
		tblist = []

		for ii, (cwl, filtername) in enumerate(zip(self.bandcenter, self.filterNameList)):
		    wave_lvf = self.filterset['wavelength']
		    resp_lvf = self.filterset[filtername]

		    # Calculate the index where response is greater than 1e-3 times the maximum response
		    indx_resp = np.where(resp_lvf > resp_lvf.max() * 1e-3)

		    # Calculate the response considering the total response
		    resp_sys = resp_lvf * res.get_total_response(wave_lvf, QE=QE, **kwargs)["response"]

		    # Create a table for the response data
		    rsptbl = Table()
		    rsptbl['index'] = [ii] * len(self.lam)
		    rsptbl['name'] = [filtername] * len(self.lam)
		    rsptbl['lam'] = self.filterset['wavelength'] * u.Angstrom
		    rsptbl['centerlam'] = cwl * u.Angstrom
		    rsptbl['bandwidth'] = self._bandwidth * u.Angstrom if isinstance(self._bandwidth, (int, float)) else self._bandwidth[-ii-1] * u.Angstrom
		    rsptbl['response'] = resp_sys
		    rsptbl['lam'].format = '.1f'
		    tblist.append(rsptbl)
		    
		    # Make zero values for both sides of the wavelength
		    rsptbl['response'][0] = 0.0
		    rsptbl['response'][-1] = 0.0

		totrsptbl = vstack(tblist)
		totrsptbl['group'] = "sdt_default"	
		self.response_table = totrsptbl.group_by("index")
		self.efficiency_table = res.get_total_response(self.lam, QE=QE, **kwargs)
		self.sky_table = res.get_sky_transmission()

	def plot_response(self, show_response=True, ax=None, **kwargs):
		if not(hasattr(self, "response_table")):
			self._calculate_response(**kwargs)

		if ax is None:
			plt.figure(figsize=(12,4))
			ax = plt.gca()
	

		if show_response:
			res.plot_table(self.efficiency_table, target="QE", lw=3, ax=ax)
			res.plot_table(self.efficiency_table, target="trans", ax=ax, lw=3, alpha=0.5)
			res.plot_table(self.efficiency_table, target="optics", ax=ax, lw=3, alpha=0.5)
			self.plot_filterset(alpha=0.25, ax=ax, add_label=False)
		else:
			self.plot_filterset(alpha=0.25, ax=ax, add_label=False)
		
		for ii, group in enumerate(self.response_table.groups):
		    res.plot_table(group, ax=ax, label=None, c=self._color[-ii-1], alpha=1)
		    if ii%2 == 0:
		        ax.text(group["centerlam"][0], 1.1*group["response"].max()*1e2, group["name"][0], horizontalalignment='center', size=10, rotation=90)
		    elif ii%2 == 1:
		        ax.text(group["centerlam"][0], 1.1*group["response"].max()*1e2, group["name"][0], horizontalalignment='center', size=10, rotation=90)
		res.plot_table(group, ax=ax, c=self._color[-ii-1], alpha=1, label="Total Response")
		ax.set_xlim(3500, 9000)
		ax.set_xticks(self._ticks)
		ax.set_ylim(0, 100)
		ax.legend(fontsize=10, loc=5)
		return ax
		
	def _get_phot_aperture(self, exposure, fwhm_seeing, optfactor=0.6731, verbose=True):
		"""_summary_
		Parameters
		----------
		exposure : float
			_description_
		fwhm_seeing : float
			_description_
		optfactor : float
			_description_

		Returns
		-------
		_type_
			_description_
		"""
		# exposure = 233000.	#	7DS Deep survey (IMS)
		# fwhm_seeing = 1.5     # [arcsec]
		fwhm_peeing = fwhm_seeing/self.optics_info["theta_pixel"]

		# How many pixels does a point source occupy?
		# Effective number of pixels for a Gaussian PSF with fwhm_seeing
		# optfactor = 0.6731

		r_arcsec = optfactor*fwhm_seeing
		r_pix = optfactor*fwhm_seeing/self.optics_info["theta_pixel"]

		aperture_diameter = 2*r_arcsec
		aperture_diameter_pix = 2*r_pix

		Npix_ptsrc = np.pi*(r_pix**2)
		Narcsec_ptsrc = np.pi*(r_arcsec**2)
		

		self._aperture_info = {}

		self.exposure = exposure
		
		self._aperture_info["aperture_multiply_factor"] = optfactor
		self._aperture_info["seeing"] = fwhm_seeing
		self._aperture_info["seeing_pix"] = fwhm_peeing
		self._aperture_info["aperture_radius_arcsec"] = r_arcsec
		self._aperture_info["aperture_radius_pix"] = r_pix
		self._aperture_info["n_aperture_pix"] = Npix_ptsrc
		self._aperture_info["n_aperture_arcsec"] = Narcsec_ptsrc
	
		if verbose:
			self.echo_aperture

	def _get_response(self, filtername):
		return self.response_table[self.response_table["name"]==filtername]["response"]

	def _get_depth_table(self, Nsigma=5):

		s = self.sky_table

		#	Empty Table
		unit_SB  = u.nW/(u.m)**2/u.sr
		unit_cntrate = u.electron / u.s

		T_sens = (QTable( 
					names=('band', 'wavelength', 'I_photo_sky', 'mag_sky', 'mag_pts'),
					dtype=(np.int16,float,float,float,float,) )
				)
		for key in T_sens.colnames:
			T_sens[key].info.format = '.4g'

		#	Iteration
		for ii, (cwl, filtername) in enumerate(zip(self.bandcenter, self.filterNameList)):
			#	Sky brightness
			wave = s['lam']*1e1 # [nm] --> [AA]
			flux = s['fnu'] # [erg/s/cm2/Hz]
			#	Filter response
			wave_sys = self.lam
			resp_sys = self._get_response(filtername)
			resp_lvf = self.filterset[filtername]

			# photon rate
			photon_rate = synth_phot(wave, flux, wave_sys, resp_sys, return_photonrate=True)
			# SB
			SB_sky = synth_phot(wave, flux, self.lam, resp_lvf)

			# photo-current or count rate
			I_photo = photon_rate * (np.pi/4*self.optics_info["d_eff"]**2) * (self.optics_info["theta_pixel"]**2)

			# noise in count per obs [e]. 
			Q_photo = (I_photo+self.optics_info["I_dark"])*self.exposure
			dQ_photo = np.sqrt(Q_photo)

			# noise in count rate [e/s]
			# read-noise (indistinguishable from signal) should be added 
			dI_photo = np.sqrt(dQ_photo**2 + self.optics_info["dQ_RN"]**2)/self.exposure

			# surface brightness limit [one pixel]

			dSB_sky = (dI_photo/I_photo)*SB_sky

			mag_sky = -2.5*np.log10(Nsigma*dSB_sky) - 48.60

			# point source limit
			dFnu = np.sqrt(self.aperture_info["n_aperture_pix"]) * dSB_sky*(self.optics_info["theta_pixel"])**2
			# dFnu = Npix_ptsrc * dSB_sky*(theta_pixel)**2
			mag_pts = -2.5*np.log10(Nsigma*dFnu) - 48.60

			# Add data to the table
			T_sens.add_row([ii, cwl, I_photo, mag_sky, mag_pts]) 

		# Put units
		T_sens['wavelength'].unit = u.um
		T_sens['I_photo_sky'].unit = unit_cntrate
		T_sens['mag_sky'].unit = u.mag
		T_sens['mag_pts'].unit = u.mag

		#	Save summary result
		outbl = Table()
		outbl['index'] = np.arange(len(T_sens))
		# outbl['name'] = [f"m{lam.value/10:g}" for lam in T_sens['wavelength']]
		outbl['name'] = self.filterNameList
		outbl['center_wavelength'] = T_sens['wavelength'].value * u.Angstrom
		outbl['fwhm'] = self._bandwidth * u.Angstrom
		outbl['min_wavelength'] = outbl['center_wavelength'] - outbl['fwhm']/2
		outbl['max_wavelength'] = outbl['center_wavelength'] + outbl['fwhm']/2
		outbl['noise_countrate'] = T_sens['I_photo_sky']
		outbl['surface_brightness'] = T_sens['mag_sky']/(u.arcsec**2)
		outbl['5sigma_depth'] = T_sens['mag_pts']
		outbl['exposure'] = self.exposure*u.second
		self.depth_table_simple = T_sens
		self.depth_table = outbl 
	
	def _get_speclite(self):
		rsptbl = self.response_table
		filterlist = self.filterNameList
		for filte in filterlist:
			#	Meta
			metadict = dict(
				group_name='sevendt',
				band_name=filte,
				exposure=self.exposure,
			)

			#	Filter Table
			fltbl = rsptbl[rsptbl['name']==filte]
			_ = speclite.filters.FilterResponse(
				wavelength = fltbl['lam'],
				response = fltbl['response'],
				meta=metadict,
			)

		#	New name for speclite class
		speclite_filterlist = [f"sevendt-{filte}" for filte in filterlist]

		#	Medium filters
		bands = speclite.filters.load_filters(*speclite_filterlist)
		self.speclite_bands = bands
	
	def get_synphot(self, data, splam=None, z=None, z0=0, show_figure=False, flux_unit="AB"):
		
		if isinstance(data, Table) and splam is None:
			spflam = data["flam"]
			splam = data["lam"]
		else:
			spflam = data
		
		rsptbl = self.response_table
		outbl = Table()
		outbl['filter'] = self.filterNameList
		outbl['lam'] = self.speclite_bands.effective_wavelengths
		outbl['bandwidth'] = np.array([rsptbl['bandwidth'][rsptbl['name']==name][0] for name in self.filterNameList])
		
		spflam, splam = self.speclite_bands.pad_spectrum(spflam, splam)
		synabsmag = self._synthesize_spectrum(spflam, splam)

		#	Shifted & Scaled spectrum
		if z!=None:
			(spflam, splam) = utils.apply_redshift_on_spectrum(spflam, splam, z, z0)
			spflam, splam = self.speclite_bands.pad_spectrum(spflam, splam)
			synappmag = self._synthesize_spectrum(spflam, splam)
		else:
			z = np.nan
			synappmag = synabsmag

		
		outbl['mag_abs'] = synabsmag
		outbl['z'] = z
		outbl['snr'] = 0.0
		outbl['mag_app'] = 0.0
		outbl['mag_obs'] = 0.0
		outbl['mag_err'] = 0.0
		outbl['fnu_obs'] = 0.0
		outbl['fnu_err'] = 0.0

		
		for ii, (mag, filtername) in enumerate(zip(synappmag, self.filterNameList)):

			(snr, m, merr) = self._calculate_magobs(mag, filtername, self.exposure)
			#	fnu [uJy]

			fnuobs = (m*u.ABmag).to(u.uJy)
			fnuerr = fnuobs/snr
			# #	flam [erg/s/cm2/AA]
			# flamobs = convert_fnu2flam(fnuobs, splam)
			# flamerr = flamobs/snr
			
			#	To the table
			outbl['snr'][ii] = snr
			outbl['mag_app'][ii] = mag
			outbl['mag_obs'][ii] = m
			outbl['mag_err'][ii] = merr
			outbl['fnu_obs'][ii] = fnuobs.value
			outbl['fnu_err'][ii] = fnuerr.value
			
			
		outbl['fnu'] = (outbl['mag_app']*u.ABmag).to(u.uJy).value

		#	Format
		for key in outbl.keys():
			if key not in ['filter']:
				outbl[key].format = '1.3f'
		#	Unit
		outbl['mag_app'].unit = u.ABmag
		outbl['mag_obs'].unit = u.ABmag
		outbl['mag_err'].unit = u.ABmag
		outbl['fnu'].unit = u.uJy
		outbl['fnu_obs'].unit = u.uJy
		outbl['fnu_err'].unit = u.uJy

		outbl['flam_obs'] = utils.convert_fnu2flam(outbl['fnu_obs'], outbl['lam'])
		outbl['flam_err'] = utils.convert_fnu2flam(outbl['fnu_err'], outbl['lam'])
		outbl['flam'] = utils.convert_fnu2flam(outbl['fnu'], outbl['lam'])

		self.synth_table = outbl

		if show_figure:
			ax = plt.gca()
			utils.plot_data(data, ax=ax, flux_unit=flux_unit)
			self.plot_syn_spectrum(ax=ax, flux_unit=flux_unit)

	def plot_syn_spectrum(self, data=None, ax = None, flux_unit="AB", add_data=False):

		if not(hasattr(self, "synth_table")) and data is not None:
			self.get_synphot(data)

		if ax is None:
			fig, ax = plt.subplots(1)

		outbl = self.synth_table

		if add_data:
			self.plot_data(outbl, ax=ax, flux_unit=flux_unit, lw=3, zorder=0, marker='.', ls='none', c='tomato', alpha=0.75, label='syn.phot')

		if flux_unit == "AB":
			flux_unit = "mag"
			ax.set_ylim([outbl['mag_obs'].max()+0.25, outbl['mag_obs'].min()-0.25])
			ax.set_ylabel(r'Brightness [AB mag]', fontsize=12)
		elif flux_unit == "fnu":
			ax.set_ylabel(r'Flux [uJy]', fontsize=12)
		elif flux_unit == "flam":
			ax.set_ylabel(r'Flux [erg/s/cm$^2$/A]', fontsize=12)

		ax.errorbar(outbl['lam'], outbl[f'{flux_unit}_obs'], xerr=outbl['bandwidth']/2, yerr=outbl[f'{flux_unit}_err'],  c='k', zorder=0, alpha=0.5, ls='none')
		scatter=ax.scatter(outbl['lam'], outbl[f'{flux_unit}_obs'], c=outbl['snr'], marker='s', edgecolors='k', s=50, label='obs')
		cbar = plt.colorbar(scatter)
		cbar.set_label('SNR', fontsize=12)

		ax.set_xlim([self.bandcenter[0]-250, self.bandcenter[-1]+250])
		
		ax.legend(loc='lower center', fontsize=10, ncol=3)
		ax.set_xlabel(r'Wavelength [$\rm \AA$]', fontsize=12)
		
		plt.tight_layout()

		return ax

	def plot_point_source_depth(self, ax=None, text=False, legend=True, add_comp=False, **kwargs):

		if ax is None:
			fig, ax = plt.subplots(1)

		T_sens = self.depth_table_simple

		for ii, filtername in enumerate(self.filterNameList):
			if ii == 0:
				ax.plot(T_sens['wavelength'][ii], T_sens['mag_pts'][ii], 'v', mec='k', c=self._color[-ii-1], ms=10, label=f'7DT ({self.exposure/3600:.1f}hr)')
			else:
				ax.plot(T_sens['wavelength'][ii], T_sens['mag_pts'][ii], 'v', mec='k', c=self._color[-ii-1], ms=10)
			if text:
				ax.text(T_sens['wavelength'][ii].value, T_sens['mag_pts'][ii].value+0.2, filtername, horizontalalignment='center', size=10)

		ax.set_xlabel(r'wavelength [$\AA$]', fontsize=20)
		ax.set_ylabel(r'Point source limits (5$\sigma$)', fontsize=20)

		# ax.set_xticks(fontsize=14)
		# ax.set_yticks(fontsize=14)

		yl, yu = ax.set_ylim()
		ax.set_ylim([yu+0.5, yl])

		# plt.xlim(wmin-(fwhm*2), wmax+(fwhm*2))
		if legend:
			ax.legend(loc='upper center', framealpha=1.0, fontsize=20)
		ax.minorticks_on()

		plt.tight_layout()
		if add_comp:
			utils.sensitivity_plot(ax=ax, **kwargs)
		return ax

	def _calculate_pointsource_snr(self, mag, filtername, exposure=None):
		s = self.sky_table

		#	Sky brightness
		wave = s['lam']*1e1 # [nm] --> [AA]
		flux = s['fnu'] # [erg/s/cm2/Hz]

		wave_sys = self.lam
		resp_sys = self._get_response(filtername)

		Naper = self.aperture_info["n_aperture_pix"]

		flux = s['fnu']

		flux_src = flux*0 + 10**(-0.4*(mag + 48.6))	# [erg/s/cm2/Hz]
		flux_sky = flux*(1e-23*1e-6) # [erg/s/cm2/Hz/arcsec2]

		photon_rate_src = synth_phot(wave, flux_src, wave_sys, resp_sys, return_photonrate=True)  # [ph/s/cm2]
		photon_rate_sky = synth_phot(wave, flux_sky, wave_sys, resp_sys, return_photonrate=True)  # [ph/s/cm2/arcsec2]

		I_photo_src = photon_rate_src * (np.pi/4*self.optics_info["d_eff"]**2)                     # [e/s] per aperture (no aperture loss)
		I_photo_sky = photon_rate_sky * (np.pi/4*self.optics_info["d_eff"]**2) * (self.optics_info["theta_pixel"]**2)  # [e/s] per pixel 

		if exposure is None:
			exposure = self.exposure
		Q_photo_src = I_photo_src * exposure
		Q_photo_sky = I_photo_sky * exposure
		Q_photo_dark = self.optics_info["I_dark"] * exposure

		snr = Q_photo_src / np.sqrt(Q_photo_src + Naper*Q_photo_sky + Naper*Q_photo_dark + Naper*self.optics_info["dQ_RN"]**2)
		return max(snr, 1e-30)
	
	def _calculate_magobs(self, mag, filtername, exposure, zperr=0.01, n=10):

		#	Signal-to-noise ratio (SNR)
		snr = self._calculate_pointsource_snr(mag=mag, filtername=filtername, exposure=exposure)
		#	SNR --> mag error
		merr0 = utils.convert_snr2magerr(snr)
		#	Random obs points
		m = utils.get_random_point(mag, merr0, n=n)
		#	Measured error
		merr = np.sqrt(merr0**2+zperr**2)

		return (snr, m, merr)

	def _synthesize_spectrum(self, spflam, splam):
	
		#	Handle NaN values
		indx_nan = np.where(np.isnan(spflam))
		indx_not_nan = np.where(~np.isnan(spflam))
		if len(indx_nan[0]) > 0:
			for nindx in indx_nan[0]:
				if nindx == 0:
					spflam[nindx] = spflam[~np.isnan(spflam)][0]
				elif nindx == len(spflam):
					spflam[nindx] = spflam[~np.isnan(spflam)][-1]
				elif (nindx != 0) & (nindx != len(spflam)) & (nindx-1 not in indx_nan[0]) & (nindx+1 not in indx_nan[0]):
					leftone = spflam[nindx-1]
					if nindx < len(spflam)-1:
						rightone = spflam[nindx+1]
					else:
						rightone = leftone

					spflam[nindx] = np.mean([leftone.value, rightone.value])*leftone.unit
				else:
					absdiff = np.abs(indx_not_nan[0]-nindx)
					closest_indx = absdiff.min()
					closest_spflam = spflam[indx_not_nan[0][closest_indx]]
					spflam[nindx] = closest_spflam

		mags = self.speclite_bands.get_ab_magnitudes(spflam, splam)
		synmag = np.array([mags[filte][0] for filte in mags.keys()])
		synmag[np.isinf(synmag)] = 999.
		synmag[np.isnan(synmag)] = 999.

		return synmag

	