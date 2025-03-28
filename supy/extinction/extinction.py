import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from dustmaps.sfd import SFDQuery
from dustmaps.planck import PlanckQuery
from dustmaps.bayestar import BayestarQuery
from .k_lambda import *

class GalacticExtinction:
    def __init__(self, reddening_map="SFD", extinction_law="cardelli", r_v=3.1):
        self.extinction_law = extinction_law
        self.r_v = r_v
        """
        Initialize the extinction calculator with the chosen dust map.

        Parameters:
        reddening_map (str): "SFD" (Schlegel+1998), "Planck" (Planck 2014), or "Bayestar" (Green+2019).
        """
        self.reddening_map = reddening_map.lower()
        if self.reddening_map == "sfd":
            self.map = SFDQuery()
        elif self.reddening_map == "planck":
            self.map = PlanckQuery()
        elif self.reddening_map == "bayestar":
            self.map = BayestarQuery()
        else:
            raise ValueError("Supported maps: 'SFD', 'Planck', 'Bayestar'")

    def get_ebv(self, ra, dec, distance=None):
        """
        Get the E(B-V) extinction value at a given RA, Dec.

        Parameters:
        ra (float): Right Ascension in degrees.
        dec (float): Declination in degrees.
        distance (float, optional): Distance in parsecs (only for Bayestar map).

        Returns:
        float: E(B-V) reddening value.
        """
        coord = SkyCoord(ra, dec, unit="deg", frame="icrs")

        if self.reddening_map == "bayestar":
            if distance is None:
                raise ValueError("Bayestar requires a distance parameter (in parsecs).")
            return self.map(coord, mode="median", d=distance / 1000.0)  # Convert to kpc

        return self.map(coord)

    def get_extinction_at_band(self, ra, dec, band="V", distance=None):
        """
        Convert E(B-V) to extinction in a specific band using standard coefficients.

        Parameters:
        ra (float): Right Ascension in degrees.
        dec (float): Declination in degrees.
        band (str): Photometric band (e.g., "U", "B", "V", "R", "I", "J", "H", "K", "G", "w").
        distance (float, optional): Distance in parsecs (only for Bayestar).

        Returns:
        float: Extinction value A_band.
        """
        ebv = self.get_ebv(ra, dec, distance)

        # Extinction coefficients from Cardelli, Clayton & Mathis (1989), Schlafly & Finkbeiner (2011)
        extinction_coefficients = {
            "U": 4.334, "B": 3.626, "V": 3.100, "R": 2.308,
            "I": 1.698, "J": 0.902, "H": 0.576, "K": 0.367,
            "G": 2.27,  # Gaia G-band extinction
            "w": 2.22   # Approximate Pan-STARRS w-band extinction
        }

        if band not in extinction_coefficients:
            raise ValueError(f"Unsupported band: {band}")

        return extinction_coefficients[band] * ebv
  
    def k_lambda(self, wavelength):
        """
        Select the appropriate extinction law for the given wavelength.

        Parameters:
        wavelength (float): Wavelength in microns.

        Returns:
        float: The extinction curve value at the given wavelength.
        """
        if self.extinction_law == "cardelli":
            return k_lambda_cardelli(wavelength)
        elif self.extinction_law == "fitzpatrick":
            return k_lambda_fitzpatrick(wavelength)
        else:
            raise ValueError(f"Unsupported extinction law: {self.extinction_law}. Use 'cardelli' or 'fitzpatrick'.")

    def get_extinction_at_wavelength(self, ra, dec, wavelength, distance=None):
        """
        Calculate the extinction for a given wavelength using the extinction law.

        Parameters:
        ra (float): Right Ascension in degrees.
        dec (float): Declination in degrees.
        wavelength (float): Wavelength in microns.
        distance (float, optional): Distance in parsecs (only for Bayestar).

        Returns:
        float: Extinction A(\lambda) at the given wavelength.
        """
        ebv = self.get_ebv(ra, dec, distance)
        k_lambda_value = self.k_lambda(wavelength)
        return self.r_v * ebv * k_lambda_value
