  
def k_lambda_cardelli(wavelength):
    """
    Calculate the extinction curve k(\lambda) for a given wavelength in microns using the Cardelli et al. (1989) law.

    Parameters:
    wavelength (float): Wavelength in microns.

    Returns:
    float: The extinction curve value at the given wavelength.
    """
    # For UV and optical wavelengths (approximation, Cardelli et al. 1989)
    if 0.3 <= wavelength <= 3.0:
        return 1.0 + 0.038 * (wavelength - 1.0) - 0.003 * (wavelength - 1.0)**2
    else:
        raise ValueError("Wavelength out of range. Supported range: 0.3 - 3.0 microns.")

def k_lambda_fitzpatrick(wavelength):
    """
    Calculate the extinction curve k(\lambda) for a given wavelength in microns using the Fitzpatrick (1999) law.

    Parameters:
    wavelength (float): Wavelength in microns.

    Returns:
    float: The extinction curve value at the given wavelength.
    """
    # Fitzpatrick (1999) extinction curve for UV-optical
    if 0.3 <= wavelength <= 3.0:
        # Standard fit for the Fitzpatrick law
        x = 1.0 / wavelength
        a = 0.574 * x ** 1.61
        b = -0.527 * x ** 1.61
        k_lambda = a + b * (wavelength - 1.0)
        return k_lambda
    else:
        raise ValueError("Wavelength out of range. Supported range: 0.3 - 3.0 microns.")
