from .mainobserver import mainObserver
import datetime
from astropy.time import Time
from astropy.coordinates import SkyCoord, AltAz
from astropy import units as u
import numpy as np
import matplotlib.pyplot as plt

class Staralt():
    
    def __init__(self, 
                 observer : mainObserver,
                 utctime : datetime or Time = None,
                 ):
        # Set the observer
        self._observer = observer
        # If no time is provided, use the current time
        if utctime is None:
            utctime = Time.now()
        if not isinstance(utctime, Time):
            utctime = Time(utctime)
        # Set the night
        self.tonight = self._set_night(utctime = utctime)
    
    @property
    def observer(self):
        return self._observer
    
    def _set_night(self, utctime : datetime or Time):
        """
        Set the night for the given time.
        
        Parameters
        ----------
        utctime : datetime or Time
            The time for which to set the night.
        """
        if not isinstance(utctime, Time):
            utctime = Time(utctime)
        class obsnight: 
            def __repr__(self):
                attrs = {name: value.iso if isinstance(value, Time) else value
                         for name, value in self.__dict__.items()}
                max_key_len = max(len(key) for key in attrs.keys())
                attrs_str = '\n'.join([f'{key:{max_key_len}}: {value}' for key, value in attrs.items()])
                return f'{self.__class__.__name__} Attributes:\n{attrs_str}'
        obsnight = obsnight()
        # Celestial information
        obsnight.sunrise_civil = self.observer.sun_risetime(utctime, horizon = 0, mode = 'next')
        obsnight.sunset_civil = self.observer.sun_settime(obsnight.sunrise_civil, mode = 'previous', horizon= 0)        
        obsnight.sunrise_nautical = self.observer.sun_risetime(obsnight.sunrise_civil, mode = 'previous', horizon= -6)
        obsnight.sunset_nautical = self.observer.sun_settime(obsnight.sunrise_civil, mode = 'previous', horizon= -6)
        obsnight.sunrise_astro = self.observer.sun_risetime(obsnight.sunrise_civil, mode = 'previous', horizon= -12)
        obsnight.sunset_astro = self.observer.sun_settime(obsnight.sunrise_civil, mode = 'previous', horizon= -12)
        obsnight.sunrise_night = self.observer.sun_risetime(obsnight.sunrise_civil, mode = 'previous', horizon= -18)
        obsnight.sunset_night = self.observer.sun_settime(obsnight.sunrise_civil, mode = 'previous', horizon= -18) 
        return obsnight

    def staralt_data(self, ra, dec, objname=None, utctime=None, target_minalt=20, target_minmoonsep=30):
        """
        Generate the data needed for the altitude plot.

        Parameters
        ----------
        ra : float
            Right Ascension of the target in degrees.
        dec : float
            Declination of the target in degrees.
        objname : str, optional
            Name of the target object.
        utctime : datetime or Time, optional
            The reference time. If None, current time is used.
        target_minalt : float, optional
            The minimum allowable altitude for observation (in degrees).
        target_minmoonsep : float, optional
            The minimum allowable moon separation (in degrees).

        Returns
        -------
        dict
            A dictionary containing all data required for plotting.
        """
        now = Time.now()
        if utctime is None:
            utctime = Time.now()
        elif not isinstance(utctime, Time):
            utctime = Time(utctime)

        # Get the sky coordinates of the target
        coord = self._get_skycoord(ra, dec)

        tonight = self.tonight
        # Define time range for plotting using Time arithmetic
        time_range_start = tonight.sunset_astro - 2*u.hour
        time_range_end = tonight.sunrise_astro + 2*u.hour

        # Generate a time axis every 5 minutes
        # We can use linspace over a Time range by converting to JD, then back to Time
        num_points = int(((time_range_end - time_range_start).sec / 60) / 5) + 1
        time_axis = time_range_start + np.linspace(0, (time_range_end - time_range_start).sec, num_points)*u.second

        # Calculate altaz for moon, sun, and target
        moon_altaz = self.observer.moon_altaz(time_axis)
        sun_altaz = self.observer.sun_altaz(time_axis)
        target_altaz = coord.transform_to(AltAz(obstime=time_axis, location=self.observer._earthlocation))

        # Calculate moon-target separation
        target_moonsep = moon_altaz.separation(target_altaz)

        # Compute colors based on criteria
        color_target_alt = ['g' if alt > target_minalt else 'r' for alt in target_altaz.alt.value]
        color_target_moonsep = ['g' if sep > target_minmoonsep else 'r' for sep in target_moonsep.value]

        # Incorporate the rule: before night start, always red
        color_target = []
        for t, alt_color, sep_color in zip(target_altaz.obstime, color_target_alt, color_target_moonsep):
            if t < tonight.sunset_night:  # before night start
                color_target.append('r')
            elif t > tonight.sunrise_night:
                color_target.append('r')
            else:
                color_target.append('r' if 'r' in (alt_color, sep_color) else 'g')

        # Convert times to ISO strings
        moon_times = [t.isoformat() for t in moon_altaz.obstime.datetime]
        sun_times = [t.isoformat() for t in sun_altaz.obstime.datetime]
        target_times = [t.isoformat() for t in target_altaz.obstime.datetime]

        # Convert altitude, separation values to lists
        moon_alts = moon_altaz.alt.value.tolist()
        sun_alts = sun_altaz.alt.value.tolist()
        target_alts = target_altaz.alt.value.tolist()
        target_moonsep_vals = target_moonsep.value.tolist()

        # Convert night times to ISO strings
        tonight_data = {
            "sunset_night": tonight.sunset_night.datetime.isoformat(),
            "sunrise_night": tonight.sunrise_night.datetime.isoformat(),
            "sunset_civil": tonight.sunset_civil.datetime.isoformat(),
            "sunrise_civil": tonight.sunrise_civil.datetime.isoformat(),
        }

        self.data_dict = {
            "objname": objname,
            "now_datetime": Time.now().datetime.isoformat(),
            "time_range_start": time_range_start.datetime.isoformat(),
            "time_range_end": time_range_end.datetime.isoformat(),
            "moon_times": moon_times,
            "moon_alts": moon_alts,
            "sun_times": sun_times,
            "sun_alts": sun_alts,
            "target_times": target_times,
            "target_alts": target_alts,
            "target_moonsep": target_moonsep_vals,
            "color_target": color_target,  # color_target is already a list of strings, so it's fine
            "tonight": tonight_data,
            "target_minalt": target_minalt,
            "target_minmoonsep": target_minmoonsep
        }

    def plot_staralt(self, data=None):
        """
        Plot the altitude data from the dictionary returned by staralt_data().

        Parameters
        ----------
        data : dict, optional
            The dictionary containing data required for plotting.
        """
        # Unpack data
        if data is None:
            data = self.data_dict

        objname = data["objname"]
        now_datetime = data["now_datetime"]
        time_range_start = data["time_range_start"]
        time_range_end = data["time_range_end"]
        moon_times = data["moon_times"]
        moon_alts = data["moon_alts"]
        sun_times = data["sun_times"]
        sun_alts = data["sun_alts"]
        target_times = data["target_times"]
        target_alts = data["target_alts"]
        target_moonsep = data["target_moonsep"]
        color_target = data["color_target"]
        tonight = data["tonight"]
        target_minalt = data["target_minalt"]
        target_minmoonsep = data["target_minmoonsep"]

        # Determine title
        if objname is None:
            titlename = 'Altitude of the Target'
        else:
            titlename = f'Altitude of the {objname}'

        # Plotting
        plt.figure(dpi=300, figsize=(10, 4))
        plt.title(titlename)

        # Plot a vertical line for current time if it's within the extended range
        if (time_range_start - datetime.timedelta(hours=3) < now_datetime < time_range_end + datetime.timedelta(hours=3)):
            plt.axvline(now_datetime, linestyle='--', c='r', label='Now')

        # Plot Moon and Sun
        plt.scatter(moon_times, moon_alts, c='b', s=10, marker='.', label='Moon')
        plt.scatter(sun_times, sun_alts, c='r', s=15, marker='.', label='Sun')

        # Plot Target
        plt.scatter(target_times, target_alts, c=color_target, s=30, marker='*', label='Target')

        # Fill nighttime regions
        plt.fill_betweenx([10, 90], tonight["sunset_night"], tonight["sunrise_night"], color='k', alpha=0.3)
        plt.fill_betweenx([10, 90], tonight["sunset_civil"], tonight["sunrise_civil"], color='k', alpha=0.1)

        # Draw vertical lines for night start/end
        plt.axvline(x=tonight["sunrise_night"], linestyle='-', c='k', linewidth=0.5)
        plt.axvline(x=tonight["sunset_night"], linestyle='-', c='k', linewidth=0.5)

        # Fill region below minimum altitude
        plt.fill_between([tonight["sunset_night"], tonight["sunrise_night"]], 0, target_minalt, color='r', alpha=0.3)

        # Add text annotations
        plt.text(tonight["sunset_night"], 93, 'Night start', fontsize=10, ha='center', va='center')
        plt.text(tonight["sunrise_night"], 93, 'Night end', fontsize=10, ha='center', va='center')
        mid_night = tonight["sunset_night"] + 0.5*(tonight["sunrise_night"] - tonight["sunset_night"])
        plt.text(mid_night, 20, 'Observation limit', fontsize=10, ha='center', va='center', fontweight='bold', c='darkred')
        plt.text(time_range_start - datetime.timedelta(hours=0.7), 85,
                 f'Current observation criteria:\n- Altitude > {target_minalt} deg\n- Moon separation > {target_minmoonsep} deg',
                 fontsize=10, ha='left', va='top', c='k', fontweight='bold')

        # Adjust plot limits and formatting
        plt.xlim(time_range_start - datetime.timedelta(hours=1), time_range_end + datetime.timedelta(hours=1))
        plt.ylim(10, 90)
        plt.legend(loc=1)
        plt.xlabel('UTC [mm-dd hh]')
        plt.ylabel('Altitude [deg]')
        plt.grid()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def _get_skycoord(self,
                     ra : str or float,
                     dec: str or float):
        """
        Parameters
        ==========
        ra : str | float = Right Ascension, if str, it should be in hms format (e.g., "10:20:30"), if float, it should be in decimal degrees
        dec : str | float = Declination, if str, it should be in dms format (e.g., "+20:30:40"), if float, it should be in decimal degrees
        
        Return
        ======
        coord : SkyCoord = SkyCoord object
        """
        
        # Check if RA and Dec are given as strings (like "10:20:30")
        if isinstance(ra, str) and isinstance(dec, str):
            # Interpret as sexagesimal format (e.g., "10:20:30", "+20:30:40")
            coord = SkyCoord(ra, dec, unit=(u.hourangle, u.deg))
        elif isinstance(ra, (float, int)) and isinstance(dec, (float, int)):
            # Interpret as decimal degrees
            coord = SkyCoord(ra * u.deg, dec * u.deg)
        else:
            raise ValueError("Unsupported RA and Dec format")
        return coord
        
# %% Example 
if __name__ == '__main__':
    import time
    A = mainObserver()
    S = Staralt(A)
    start = time.time()
    S.staralt_data(ra = 20.5243, dec = -20.245, objname = 'ABC', target_minalt = 30)
    S.plot_staralt()
    print(f'Elapsed time: {time.time() - start:.2f} s')
# %%
