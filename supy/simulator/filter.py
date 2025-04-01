import numpy as np
import matplotlib.pyplot as plt
import glob
from astropy.table import Table
from . import utils
from pathlib import Path

from .response import plot_table
from ..const import REFDATA_DIR

class Filter:

	_refdata = Path(REFDATA_DIR)

	def __init__(self, filterset="default", verbose=False, **kwargs):
		if filterset.lower() == "sdss":
			if verbose: 
				print("Adopt the SDSS filterset")
			self.sdss_filterset()
		elif filterset.lower() == "tophat":
			if verbose: 
				print("Adopt the tophat filterset")
			self.tophat_filterset(**kwargs)
		elif filterset.lower() == "spherex":
			if verbose: 
				print("Adopt the SphereX filterset")
			self.tophat_filterset(**kwargs)
		elif filterset.lower() == "default":
			if verbose: 
				print("Adopt the default 7DT filterset")
			self.default_filterset()

	@classmethod
	def tophat_filterset(self, bandmin, bandmax, bandwidth, bandstep, bandrsp, lammin=1000, lammax=10000, lamres=1000):
		lam = np.arange(bandmin, bandmax, bandstep)
		wave = np.linspace(lammin, lammax, lamres)

		#	Create filter_set definition
		filter_set = {
			'wavelength': wave
			}

		filterNameList = []
		for ii, wl_cen in enumerate(lam):
			rsp = utils.tophat_trans(wave, center=wl_cen, fwhm=bandwidth)*bandrsp
			filtername = f'm{wl_cen:g}'
			filter_set.update({filtername: rsp})
			indx_resp = np.where(rsp>rsp.max()*1e-3)
			filterNameList.append(filtername)

		#	ticks
		step = 500
		ticks = np.arange(round(lam.min(), -2)-step, round(lam.max(), -2)+step, step)
		self.filterset = Table(filter_set)
		self.filterNameList = filterNameList
		self.lam = wave
		self._lammin = lammin
		self._lammax = lammax
		self._lamres = lamres
		self.bandcenter = lam
		self._bandstep = bandstep
		self._bandwidth = bandwidth
		self._color = utils.makeSpecColors(len(lam))
		self._ticks = ticks
		return filter_set

	@classmethod
	def default_filterset(self):
		#	Subsequent filter info [AA]
		bandmin=4000
		bandmax=9000
		bandwidth=250
		bandstep=125
		#	Maximum transmission of each filters
		bandrsp=0.95
		#	Wavelength bin [AA]
		lammin=1000
		lammax=10000
		lamres=1000
		return self.tophat_filterset(bandmin=bandmin, bandmax=bandmax, bandwidth=bandwidth, bandstep=bandstep, bandrsp=bandrsp, lammin=lammin, lammax=lammax, lamres=lamres)
	
	@classmethod
	def sdss_filterset(self):
		#	Filter Table List
		path = str(self._refdata)+'/SDSS/Chroma*'
		sdsslist = sorted(glob.glob(path))
		#	Re-arange the order (ugriz)
		sdsslist = [sdsslist[3], sdsslist[0], sdsslist[2], sdsslist[1], sdsslist[4]]

		#	Create filter_set definition
		filter_set = {
			'wavelength': 0
			}

		cwl = []
		filterNameList = []
		bandwidth = []
		for ss, sdss in enumerate(sdsslist):
			#	Read SDSS filter transmission table
			intbl = Table.read(sdss, format="ascii")
			#	Get filter name
			filte = sdss.split('_')[1][0]

			if len(intbl) >= 1801:
				#	[nm] --> [Angstrom]
				intbl['lam'] = intbl['col1']*10
				#	Wavelength resolution
				lamres = np.mean(intbl['lam'][1:] - intbl['lam'][:-1])
				#	Wavelength min & max
				lammin = intbl['lam'].min()
				lammax = intbl['lam'].max()
			else:
				reftbl = Table.read(sdsslist[0], format="ascii")
				rsp = np.interp(reftbl['col1'], intbl['col1'], intbl['col2'])
				intbl = Table()
				#	[nm] --> [Angstrom]
				intbl['lam'] = reftbl['col1']*10
				intbl['col2'] = rsp
				#	Wavelength resolution
				lamres = np.mean(intbl['lam'][1:] - intbl['lam'][:-1])
				#	Wavelength min & max
				lammin = intbl['lam'].min()
				lammax = intbl['lam'].max()

			#	Effective wavelength
			# indx_eff = np.where(intbl['col2']>0.5)
			# cwl.append(np.sum(intbl['lam'][indx_eff]*intbl['col2'][indx_eff]*lamres)/np.sum(intbl['col2'][indx_eff]*lamres))
			cwl.append(np.sum(intbl['lam']*intbl['col2']*lamres)/np.sum(intbl['col2']*lamres))
			filterNameList.append(filte)
			filter_set.update({filte: intbl['col2']})

			#	Half Max
			hm = intbl['col2'].max()/2
			#	Left, Right index
			indx_left = np.where(intbl['lam']<np.sum(intbl['lam']*intbl['col2']*lamres)/np.sum(intbl['col2']*lamres))
			indx_right = np.where(intbl['lam']>np.sum(intbl['lam']*intbl['col2']*lamres)/np.sum(intbl['col2']*lamres))
			#	FWHM
			fwhm = np.interp(hm, intbl['col2'][indx_right], intbl['lam'][indx_right],) - np.interp(hm, intbl['col2'][indx_left], intbl['lam'][indx_left],)
			bandwidth.append(fwhm/2)
		# bandwidth = np.array(bandwidth)
		#	Forced value
		#	https://www.researchgate.net/figure/SDSS-FILTER-CHARACTERISTICS-AND-PHOTOMETRIC-SENSITIVITY-14-AIR-MASSES_tbl2_2655119
		bandwidth = np.array(
			[
				560,
				1377,
				1371,
				1510,
				940,
			]
		)
		cwl = np.array(cwl)

		step = 500
		ticks = np.arange(round(intbl['lam'].min(), -3)-step, round(intbl['lam'].max(), -3)+step, step)

		
		filter_set['wavelength'] = np.array(intbl['lam'])
		self.filterset = Table(filter_set)
		self.filterNameList = filterNameList
		self.lam = intbl['lam']
		self._lammin = lammin
		self._lammax = lammax
		self._lamres = lamres
		self.bandcenter = cwl
		self._bandstep = 0
		self._bandwidth = bandwidth
		self._color = utils.makeSpecColors(len(cwl))
		self._ticks = ticks
		return filter_set

	def plot_filterset(self, ax=None, add_label=True, **kwargs):

		filterset = self.filterset
		
		if ax is None:
			plt.figure(figsize=(12, 4))
			ax = plt.gca()
		
		#	Wavelength
		lam = filterset['wavelength']
		for ii, filtername in enumerate(self.filterNameList):
			#	Central wavelength
			cwl = self.bandcenter[ii]
			#	Response [%]
			rsp = filterset[filtername]*1e2
			#	Cut the tails of curve
			indx_rsp = np.where(rsp>rsp.max()*1e-3)
			#	Plot
			ax.plot(lam[indx_rsp], rsp[indx_rsp], c=self._color[-ii-1], lw=3, **kwargs)
			if add_label:
				if ii%2 == 0:
					ax.text(cwl, 100, filtername, horizontalalignment='center', size=9)
				elif ii%2 == 1:
					ax.text(cwl, 107.5, filtername, horizontalalignment='center', size=9)
		#	Plot Setting
		# step = 500
		# xticks = np.arange(round(filterset['cwl'].min(), -2)-step, round(filterset['cwl'].max(), -2)+step, step)
		# plt.xticks(xticks)
		ax.set_xticks(self._ticks)
		ax.grid(axis='y', ls='-', c='silver', lw=1, alpha=0.5)
		ax.set_ylim(0, 1.15*1e2)
		ax.set_xlabel(r'Wavelength [$\rm \AA$]', fontsize=20)
		ax.set_ylabel('Transmission [%]', fontsize=20)
		ax.minorticks_on()
		plt.tight_layout()
		return ax
	