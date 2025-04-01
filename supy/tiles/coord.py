from astropy.coordinates import SkyCoord
import astropy.units as u
import re

def parse_coordinates(coords, format=None, output_format='degrees'):
    """
    Parse astronomical coordinates and return RA and Dec in the specified format.
    
    Parameters:
    -----------
    coords : str, tuple, list, or dict
        The coordinates to parse.
    format : str, optional
        Force a specific format interpretation ('sexagesimal', 'degrees', 'hmsdms', 'j2000').
    output_format : str, optional
        Output format ('degrees', 'hmsdms', or 'j2000'). Default is 'degrees'.
    
    Returns:
    --------
    tuple
        (RA, Dec) in the requested format.
    """
    if isinstance(coords, dict):
        ra_str, dec_str = str(coords.get('ra', '')).strip(), str(coords.get('dec', '')).strip()
    elif isinstance(coords, (list, tuple)) and len(coords) >= 2:
        ra_str, dec_str = str(coords[0]).strip(), str(coords[1]).strip()
    elif isinstance(coords, str):
        if coords.startswith('J') and format != 'sexagesimal' and format != 'degrees':
            return _parse_j2000(coords, output_format)
        parts = re.split(r'[ ,]+', coords.strip(), maxsplit=1)
        if len(parts) < 2:
            raise ValueError(f"Cannot parse RA and Dec from '{coords}'.")
        ra_str, dec_str = parts[0], parts[1]
    else:
        raise ValueError("Coordinates must be a string, tuple, list, or dictionary")
    
    if format is None:
        format = 'sexagesimal' if _is_sexagesimal(ra_str) else 'degrees'
    
    if format == 'sexagesimal' or format == 'hmsdms':
        ra_unit = u.hourangle if ':' in ra_str or 'h' in ra_str else u.deg
        coord = SkyCoord(ra_str, dec_str, unit=(ra_unit, u.deg))
    elif format == 'degrees':
        coord = SkyCoord(float(ra_str) * u.deg, float(dec_str) * u.deg)
    elif format == 'j2000':
        return _parse_j2000(ra_str, output_format)
    else:
        raise ValueError(f"Unknown format: {format}")
    
    if output_format == 'degrees':
        return coord.ra.deg, coord.dec.deg
    elif output_format == 'hmsdms':
        return coord.ra.to_string(unit=u.hourangle, sep=':'), coord.dec.to_string(unit=u.deg, sep=':')
    elif output_format == 'j2000':
        return _format_j2000(coord)
    else:
        raise ValueError(f"Unknown output format: {output_format}")

def _is_sexagesimal(coord_str):
    """Check if a coordinate string is in sexagesimal format."""
    # Check for hour/degree:minute:second format
    if ':' in coord_str:
        return True
    # Check for hour/degree minute second format
    if re.search(r'\d+\s+\d+\s+\d+(\.\d*)?', coord_str):
        return True
    # Check for HMS/DMS format
    if any(char in coord_str for char in 'hmsHMSdDM°\'′″"'):
        return True
    return False
    
def _parse_j2000(j2000_str, output_format):
    raw = j2000_str[1:]
    split_idx = next((i for i, c in enumerate(raw) if i > 5 and c in ['+', '-']), None)
    if split_idx is None:
        raise ValueError(f"Cannot parse RA/Dec from {j2000_str}")
    
    ra_h, ra_m, ra_s = int(raw[:2]), int(raw[2:4]), float(raw[4:split_idx])
    dec_sign, dec_raw = (-1 if raw[split_idx] == '-' else 1), raw[split_idx+1:]
    dec_d, dec_m, dec_s = int(dec_raw[:2]), int(dec_raw[2:4]), float(dec_raw[4:])
    
    coord = SkyCoord(f"{ra_h}h{ra_m}m{ra_s:.2f}s {dec_sign*dec_d}d{dec_m}m{dec_s:.2f}s", frame='icrs')
    
    if output_format == 'degrees':
        return coord.ra.deg, coord.dec.deg
    elif output_format == 'hmsdms':
        return coord.ra.to_string(unit=u.hourangle, sep=':'), coord.dec.to_string(unit=u.deg, sep=':')
    elif output_format == 'j2000':
        return _format_j2000(coord)

def _format_j2000(coord):
    ra_hms = coord.ra.hms
    dec_dms = coord.dec.dms
    sign = '+' if dec_dms.d >= 0 else '-'
    return f"J{int(ra_hms.h):02d}{int(ra_hms.m):02d}{ra_hms.s:.2f}{sign}{abs(int(dec_dms.d)):02d}{int(dec_dms.m):02d}{abs(dec_dms.s):.2f}"
