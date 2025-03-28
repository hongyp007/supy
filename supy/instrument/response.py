from pathlib import Path
import numpy as np
from astropy.table import Table
import astropy.units as u
import astropy.constants as c
from scipy.optimize import curve_fit
from . import utils
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

def get_QE_factor(target="default"):
	if target=="default":
		return CMOS_IMX455_QE()
	elif target=="CMOS_IMX455_QE_forum":
		return CMOS_IMX455_QE_forum()
	elif target=="Hamamtsu":
		return CCD_Hamamtsu_QE()
	else:
		return np.nan

def CCD_Hamamtsu_QE():
	# QE table of Gemini GMOS-N Hamamatsu CCD
	T_comp = Table.read('http://www.gemini.edu/sciops/instruments/gmos/gmos_n_ccd_hamamatsu_sc.txt', format='ascii.no_header', names=('wavelength', 'QE'))
	# T_comp['wavelength'] = T_comp['wavelength'].astype(float) * 1e-3
	T_comp['wavelength'].name = "lam"
	T_comp['lam'] = T_comp['lam'].astype(float) * 1e1
	T_comp['lam'].unit = u.Angstrom
	T_comp['lam'].format = '8.4f'
	return T_comp


def CMOS_IMX455_QE(path_table='QE.csv'):
	T_qe = Table.read(REFDATA_DIR+f"/qe/{path_table}", format="ascii")
	T_qe['lam'] = T_qe['wave'] * 1e1
	T_qe['lam'].unit = u.Angstrom
	T_qe['lam'].format = '8.4f'
	return T_qe

def CMOS_IMX455_QE_forum(path_table='sony.imx455.qhy600.dat'):
	T_qe = Table.read(REFDATA_DIR+f"/qe/{path_table}", format="ascii")
	T_qe['lam'] = T_qe['lam'] * 1e1
	T_qe['lam'].unit = u.Angstrom
	T_qe['lam'].format = '8.4f'
	T_qe['QE'] = T_qe['qe']*1e-2

	#	Curve-fit interpolation
	lam_optics = T_qe['lam'].value
	total_optics = T_qe['QE']
	popt, pcov = curve_fit(utils.func_linear, lam_optics[:2], total_optics[:2])
	xdata = np.arange(1000, 4000, 50)
	ydata = utils.func_linear(xdata, *popt)
	nx = np.append(xdata, lam_optics)
	ny = np.append(ydata, total_optics)
	ny[ny<0] = 0

	nT_qe = Table()
	nT_qe['lam'] = nx
	nT_qe['lam'].unit = u.Angstrom
	nT_qe['lam'].format = '8.4f'
	nT_qe['QE'] = ny

	return nT_qe

def get_sky_transmission(path_table='skytable.fits', apply_filter=True, smooth_fractor=10):
	table = Table.read(REFDATA_DIR+f"/{path_table}")

	lam = table['lam']*u.nm
	#	Photon Rate [ph/s/m2/micron/arcsec2]
	I = table['flux']*u.photon/u.second/(u.m**2)/u.um/(u.arcsec**2)
	flam = (c.h*c.c/lam)*I/(u.photon/u.arcsec**2)
	flam = flam.to(u.erg/u.second/u.cm**2/u.Angstrom)
	fnu = utils.convert_flam2fnu(flam, lam)
	abmag = fnu.to(u.ABmag)
	sky_table = Table()
	sky_table['lam'] = lam*10
	sky_table['lam'].unit = u.Angstrom
	sky_table['flam'] = flam
	sky_table['fnu'] = fnu
	sky_table['abmag'] = abmag
	sky_table['trans'] = table['trans']
	
	if apply_filter:
		sky_table['trans'] = gaussian_filter(sky_table['trans'], smooth_fractor)

	return sky_table
		
def get_optics(path_table='optics.efficiency.dr350.dr500.csv'):
	optbl = Table.read(REFDATA_DIR+f"/{path_table}", format="ascii")

	optbl['nm'].unit = u.nanometer
	optbl['lam'] = optbl['nm']*10
	optbl['lam'].unit =u.Angstrom
	optbl.remove_column("Total")
	optbl['total'].name = "optics"

	popt, pcov = curve_fit(utils.func_linear, optbl['lam'][:2], optbl['optics'][:2])
	
	temp = utils.func_linear(3000, *popt)
	optbl.add_row([300, np.nan, np.nan, np.nan, np.nan, np.nan, temp, 3000])
	optbl.sort("nm")
	return optbl

def get_total_response(wavelengths, QE="default", **kwargs):
	qe = get_QE_factor(QE)
	sky = get_sky_transmission(**kwargs)
	opt = get_optics()
	r1 = np.interp(wavelengths, qe["lam"], qe["QE"])
	r2 = np.interp(wavelengths, sky["lam"], sky["trans"])
	r3 = np.interp(wavelengths, opt["lam"], opt["optics"])
	rate = r1*r2*r3
	table = Table()
	table["lam"] = wavelengths
	table["lam"].unit = u.Angstrom
	table["QE"] = r1
	table["trans"] = r2
	table["optics"] = r3
	table["response"] = rate
	table["percent"] = table["response"]*100
	table["percent"].unit = u.percent
	return table

def plot_table(table=None, target=None, add_7ds_range=False, ax=None, scale=100, **kwargs):

	if ax is None:
		plt.figure(figsize=(12,4))
		ax = plt.gca()
	
	exist_label=kwargs.get("label", False)
	
	if "response" in table.keys() and target is None:
		ax.plot(table['lam'][table['response']>0], table['response'][table['response']>0]*1e2, label=kwargs.pop("label", "Response"), **kwargs)
		ax.set_ylabel('Response [%]', fontsize=15)
		ax.set_ylim(-10, 100.0)
	elif target=="brightness":
		ax.plot(table['lam'], table['abmag'], c=kwargs.pop("color",'dodgerblue'), label=kwargs.pop("label", "Sky brightness"),**kwargs)
		ax.set_ylabel(r'$SB_\nu$ [$mag/arcsec^2$]', fontsize=15)
		ax.set_ylim(24,14)
	elif ("QE" in table.keys() and target is None) or target is "QE":
		ax.plot(table['lam'][table['QE']>0], table['QE'][table['QE']>0]*1e2, c=kwargs.pop("color",'silver'), label=kwargs.pop("label", "Quentum efficiency"), **kwargs)
		ax.set_ylabel('Quentum Efficiency [%]', fontsize=15)
		ax.set_ylim(-10, 100.0)
	elif ("trans" in table.keys() and target is None) or target is "trans":
		ax.plot(table['lam'][table['trans']>0], table['trans'][table['trans']!=0]*1e2, c=kwargs.pop("color",'dodgerblue'), label=kwargs.pop("label", "Sky transmission"), **kwargs)
		ax.set_ylabel('Transmission [%]', fontsize=15)
		ax.set_ylim(-10, 100.0)
	elif ("optics" in table.keys() and target is None)  or target is "optics":
		ax.plot(table['lam'][table['optics']>0], table['optics'][table['optics']>0]*1e2, c=kwargs.pop("color",'purple'), label=kwargs.pop("label", "Optics system"), **kwargs)
		ax.set_ylabel('Efficiency [%]', fontsize=15)
		ax.set_ylim(-10, 100.0)
	else:
		ax.plot(table['lam'][table[target]>0], table[target][table[target]>0]*scale, label=kwargs.pop("label", "Response"), **kwargs)
		return

	#	7DS range
	if add_7ds_range:
		ax.axvspan(4000, 9000, facecolor='silver', alpha=0.5, label='7DT range')

	ax.set_xlim(3000, 11000)
	ax.set_xlabel(r'Wavelength [$\rm \AA$]', fontsize=15)
	
	if exist_label:
		ax.legend(loc='upper right')
	ax.minorticks_on()
	plt.tight_layout()
	return ax


