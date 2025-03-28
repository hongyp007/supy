import numpy as np
import astropy.units as u
import astropy.constants as c

# Constants
c_ums = 3e14                  # c in um/s
c = 3e8                       # m/s
h = 6.626e-34                 # Planck constant   [J/Hz]
k = 1.38e-23                  # Boltzman constant [J/K]

rad2arcsec = (180/np.pi*3600) # 206265 arcsec
arcsec2rad = 1/rad2arcsec

#	Unit
lamunit = u.Angstrom
flamunit = u.erg/u.second/u.cm**2/u.Angstrom
#

filterlist_med25nm = (
	('m4000', '25nm', 'med25nm'),
	('m4250', '25nm', 'med25nm'),
	('m4500', '25nm', 'med25nm'),
	('m4750', '25nm', 'med25nm'),
	('m5000', '25nm', 'med25nm'),
	('m5250', '25nm', 'med25nm'),
	('m5500', '25nm', 'med25nm'),
	('m5750', '25nm', 'med25nm'),
	('m6000', '25nm', 'med25nm'),
	('m6250', '25nm', 'med25nm'),
	('m6500', '25nm', 'med25nm'),
	('m6750', '25nm', 'med25nm'),
	('m7000', '25nm', 'med25nm'),
	('m7250', '25nm', 'med25nm'),
	('m7500', '25nm', 'med25nm'),
	('m7750', '25nm', 'med25nm'),
	('m8000', '25nm', 'med25nm'),
	('m8250', '25nm', 'med25nm'),
	('m8500', '25nm', 'med25nm'),
	('m8750', '25nm', 'med25nm'),
)

filterlist_griz = (
	# ('u', 'broad', 'broad'),
	('g', 'broad', 'broad'),
	('r', 'broad', 'broad'),
	('i', 'broad', 'broad'),
	('z', 'broad', 'broad'),
)

filterlist_ugriz = (
	('u', 'broad', 'broad'),
	('g', 'broad', 'broad'),
	('r', 'broad', 'broad'),
	('i', 'broad', 'broad'),
	('z', 'broad', 'broad'),
)

# http://svo2.cab.inta-csic.es/svo/theory/fps/index.php?mode=browse&gname=SLOAN&asttype=
bandwidtharr_broad_ugriz = np.array([
	#	ugriz
	540.97,
	1064.68,
	1055.51,
	1102.57,
	1164.01,
	#	u'g'r'i'z'
	# 563.56,
	# 1264.52,
	# 1253.71,
	# 1478.93,
	# 4306.72,
])

bandwidtharr_broad_griz = np.array([
	#	griz
	1064.68,
	1055.51,
	1102.57,
	1164.01,
])

bandwidtharr_med25nm = np.array([250]*20)