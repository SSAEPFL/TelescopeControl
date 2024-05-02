from typing import Protocol, List
from enum import Enum
from hardware import Hardware, Camera, Mount, Focuser, FilterWheel
import skyfield
from skyfield.api import EarthSatellite, load, wgs84
from skyfield.timelib import Time
import logging
import helperfunctions
from pathlib import Path
from astropy.coordinates import SkyCoord
import numpy as np
from datetime import datetime, timedelta
from time import sleep
import cv2
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score


class StationStatus(Enum):
    UNKNOWN = "Unknown"
    OPERATIONAL = "Operational"
    INOPERATIONAL = "Inoperational"
    OBSERVING = "Observing"


class Station(Protocol):
    # TODO: recheck use of protocol
    """This protocol models a station, which has a location an ID and hardware."""
    station_id: str  # The ID of the station
    # The position of the station on Earth
    position: skyfield.toposlib.GeographicPosition
    # The operational status of the station, can be "Unknown", "Operational", "Inoperational" or "Observing"
    status: StationStatus
    hardware: List[Hardware]  # The hardware of the station
    datafolder: str  # The folder where the data is stored

    def __init__(self, station_id: str, position: skyfield.toposlib.GeographicPosition, status=StationStatus.UNKNOWN):
        self.station_id = station_id
        self.position = position
        self.status = status
        self.hardware = None

    def add_hardware(self, hardware: Hardware) -> None:
        self.hardware.append(hardware)

    def initialize_hardware(self) -> None:
        """Initializes the hardware of the station."""
        for hardware in self.hardware:
            hardware.initialize()

    def get_measurements(self) -> None:
        """Extracts the measurement from the observations done by the station."""
        pass


class OpticalTrackingStation(Station):
    """This class models an optical tracking station, which has a location, an ID and specific hardware. It has specifically 2 cameras, a primary and a secondary one mounted on a telescope. The primary camera is used for fine tracking and data-acquisition and the secondary one is used for coarse tracking. The primary camera is attached to the optical tube that can be focused. The secondary camera is attached on the tube and has a wider field of view. Both cameras are mounted on the Mount object."""

    def __init__(self, station_id: str,
                 position: skyfield.toposlib.GeographicPosition,
                 primary_camera: Camera,
                 secondary_camera: Camera,
                 mount: Mount,
                 focuser: Focuser,
                 filterwheel: FilterWheel,
                 datafolder=None):
        # Log setup
        self._logger = logging.getLogger(
            "station.OpticalTrackingStation."+station_id)
        self._logger.setLevel(logging.DEBUG)
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        # File handler
        fh = logging.FileHandler("station.log")
        fh.setLevel(logging.DEBUG)

        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        fh.setFormatter(formatter)

        # Adding handlers
        self._logger.addHandler(ch)
        self._logger.addHandler(fh)

        # Start of constructor
        self._logger.debug("Creating new instance of optical tracking station with id: " + station_id + " and position: " +
                           str(position) + " and hardware: " + str(primary_camera) + ", " + str(secondary_camera) + ", " + str(mount) + ", " + str(focuser) + ".")
        self.station_id = station_id
        self.position = position

        self.primary_camera = primary_camera
        self.secondary_camera = secondary_camera
        self.mount = mount
        self.focuser = focuser
        self.filterwheel = filterwheel

        self.status = StationStatus.UNKNOWN
        self.hardware = [self.primary_camera,
                         self.secondary_camera, self.mount, self.focuser]

        if datafolder is None:
            self.datafolder = Path(__file__) / "data"
        else:
            self.datafolder = datafolder

    def __str__(self):
        return "OpticalTrackingStation: " + self.station_id + "."

    # Loggers with name
    def _log_debug(self, message: str, **kwargs) -> None:
        self._logger.debug(self.station_id + ": " + message, **kwargs)

    def _log_info(self, message: str, **kwargs) -> None:
        self._logger.info(self.station_id + ": " + message, **kwargs)

    def _log_warning(self, message: str, **kwargs) -> None:
        self._logger.warning(self.station_id + ": " + message, **kwargs)

    def _log_error(self, message: str, **kwargs) -> None:
        self._logger.error(self.station_id + ": " + message, **kwargs)

    def initialize_hardware(self) -> None:
        """Initializes the hardware of the station."""
        self._log_info("Initializing hardware")
        for hardware in self.hardware:
            hardware.initialize()

    def get_measurements(self) -> None:
        """Extracts the measurement from the observations done by the station."""
        self._log_info("Getting measurements")
        pass

    def can_observe(self, satellite: EarthSatellite, time) -> bool:
        """Returns whether the station can observe a given satellite."""
        self._log_info("Checking if station can observe satellite: " +
                       satellite.model.intldesg, " at time: " + time + ".")
        # Computing alt and az of satellite using helperfunctions
        alt, az = helperfunctions.get_sat_altaz_position(
            satellite, self.position, time)
        self._log_debug("Computed following Altitude: " + alt +
                        ", azimuth: " + az + ", distance: " + distance + ".")

        # Loading the ephemerides
        eph = load('datafiles/de440s.bsp')

        # Checking if the satellite is above the horizon and sunlit
        if alt.degrees > 10 and satellite.at(t).is_sunlit(eph):
            self._log_info("Station can observe satellite: " +
                           satellite.model.intldesg, " at time: " + time + ".")
            return True
        else:
            self._log_info("Station cannot observe satellite: " +
                           satellite.model.intldesg, " at time: " + time + ".")
            return False

    def display_image(self, image):
        """Displays an image on the screen using cv2."""
        cv2.imshow('Tracked Particles', frame)
        cv2.waitKey(1)

    def observe(self, satellite: EarthSatellite, time: Time) -> None:
        """Observes a given satellite at a given time."""
        self._log_info("Observing satellite: " +
                       satellite.model.intldesg + " at time: " + str(time) + ".")
        # loading timescale
        ts = load.timescale()

        # Updating status
        self.status = StationStatus.OBSERVING

        # Slewing to the predicted position of the satellite at time �
        self._log_info("Trying to slew to predicted position of satellite: " +
                       satellite.model.intldesg + " at time: " + str(time) + ".")
        ra, dec = helperfunctions.get_sat_radec_position(
            satellite, self.position, time)
        position = SkyCoord(ra=ra._degrees, dec=dec._degrees,
                            frame='icrs', unit='deg')
        try:
            self.mount.slew_to(position)
        except:
            self._log_error("Slewing to predicted position of satellite: " +
                            satellite.model.intldesg + " at time: " + str(time) + " failed.")
            raise Exception("Slewing to predicted position of satellite: " +
                            satellite.model.intldesg + " at time: " + str(time) + " failed.")

        while self.mount.is_slewing():
            sleep(0.1)

        # TODO: change this function to compute the time needed for the satellite to cross the field of view of the primary camera and ajust accordingly
        # Waiting for the time of observation minus 5 seconds
        self._log_info("Waiting for the time of observation minus 5 seconds")
        t = ts.now()

        while t.utc < (time - timedelta(seconds=5)).utc:
            t = ts.now()
            sleep(0.1)

        # Taking images with the primary camera with a given exposure time during 15 seconds
        self._log_info(
            "Taking images with the primary camera during 15 seconds")

        # starting loop exposure during 15 seconds
        self.primary_camera.activate_loop_mode()

        # exposing during 15 seconds, displaying images on the screen
        self.primary_camera.start_exposure()
        while t.utc < (time + timedelta(seconds=10)).utc:
            t = ts.now()
            if False:
                self.display_image(self.primary_camera.get_last_image())
            sleep(0.1)

        self.primary_camera.stop_exposure()

        # deactivating loop mode
        self.primary_camera.deactivate_loop_mode()

        # collecting images
        self._log_info("Collecting images")
        images = self.primary_camera.get_images()

        # saving images
        self._log_info("Saving images")
        # TODO: check this
        # for image in images:
        #     image.save(self.datafolder / "images" / self.station_id /
        #                "primary_camera" / str(time) + ".fits")

        self._log_info("Observation completed")

        #

    def track(self, satellite: EarthSatellite, start_time: Time, end_time: Time) -> None:
        """Tracks a given satellite at a given time."""
        self._log_info("Tracking satellite: " +
                       satellite.model.intldesg + " at time: " + str(start_time) + ".")
        # loading timescale
        ts = load.timescale()

        # Updating status
        self.status = StationStatus.OBSERVING

        # Finding where the satellite is in the sky at time
        ra, dec = helperfunctions.get_sat_radec_position(
            satellite, self.position, start_time)

        # Casting angles into SkyCoord
        position = SkyCoord(ra=ra._degrees, dec=dec._degrees,
                            frame='icrs', unit='deg')
        # slewing to the given position
        self.mount.slew_to(position)

        self._log_info("Waiting for mount to finish slewing.")

        # waiting for the mount to finish slewing
        while self.mount.is_slewing():
            sleep(0.1)

        self._log_info("Waiting for satellite")

        t = ts.now()
        # waiting for the satellite to be in the field of view
        # Currently does nothing
        while not helperfunctions.is_in_fov(satellite, self.position, self.primary_camera, t):
            sleep(0.1)
            t = ts.now()

        while t.utc < start_time.utc:
            t = ts.now()-timedelta(seconds=self.mount.command_delay)
            sleep(0.1)

        self._log_info("Satellite predicted to be in field of view")
        self._log_debug("Starting exposure")

        # starting exposure
        self.primary_camera.activate_loop_mode()
        self.primary_camera.start_exposure()

        # start tracking #TODO: implement further tracking methods

        self._log_info("Starting tracking")
        while t.utc < end_time.utc:
            t = ts.now()
            t_command = t+timedelta(seconds=self.mount.command_delay)
            ra_rate, dec_rate = helperfunctions.get_sat_radec_velocity(
                satellite, self.position, t_command)
            self.mount.set_slew_rate(ra_rate, dec_rate)
            sleep(self.mount.update_rate)

        self._log_info("Stopping tracking")
        self.mount.set_slew_rate(0, 0)
        # stopping exposure by deactivating loop mode
        self._log_debug("Stopping exposures")

        # deactivating loop mode
        self.primary_camera.deactivate_loop_mode()

        # collecting images
        self._log_info("Collecting images")
        self._images = self.primary_camera.get_images()

        # saving images #TODO: check this
        self._log_info("Saving images")
        # for image in images:
        #    image.save(self.datafolder / "images_" / self.station_id /
        #              "_primary_camera_" / str(time) + ".fits")
        self._log_info("Reinitializing Mount")
        self.mount.initialize()
        # Updating status
        self.status = StationStatus.OPERATIONAL

    def autofocusAllWheels(self):
        if self.filterwheel is None:
            raise RuntimeError("No filterwheel in the station")
        focus_points = {}
        for filter in self.filterwheel.get_names():
            self.filterwheel.select_filter(filter)
            focus_points[filter] = self.autofocus()

    def autofocus(self):
        self._log_info("Attempting Autofocus")
        if self.focuser is None:
            raise RuntimeError("No focuser in the station")
        if self.camera is None:
            raise RuntimeError("No camera in the station")
        # move N_steps higher and measure FWHM
        self.autofocus_properties = {
            'N_steps_positive': 5,
            'N_steps_negative': 5,
            'step_width': 20,
            'gain': 200,
            'exp_time': 1,
        }
        positions = []
        FWHMs = []
        FWHM_std_devs = []
        for n in range(self.autofocus_properties['N_steps_positive']):
            self.focuser.go_to_position(self.focuser.get_position() + self.autofocus_properties['step_width'])
            positions.append( self.focuser.get_position() )
            sleep(0.5)
            img = self.camera.take_image(self.autofocus_properties['gain'], self.autofocus_properties['exp_time'])
            fwhm, fwhm_std_dev = helperfunctions.get_FWHM(img)
            FWHMs.append( fwhm )
            FWHM_std_devs.append( fwhm_std_dev )
            self._log_info("Step {}, FWHM={} pm {}".format(n, fwhm, fwhm_std_dev))

        # go to expected minimum and measure FWHM
        fun = lambda x, p: p[0]*(x-p[1])**2 + p[2]
        p0 = [1, 1, 4]
        popt, pcov = curve_fit(fun, np.array(positions), FWHMs, p0=p0)
        min_pos = popt[1]
        self.focuser.go_to_position(min_pos)
        sleep(0.5)
        img = self.camera.take_image(self.autofocus_properties['gain'], self.autofocus_properties['exp_time'])
        fwhm, fwhm_std_dev = helperfunctions.get_FWHM(img) 
        FWHMs.append( fwhm )
        FWHM_std_devs.append( fwhm_std_dev )


        # move N_steps_negative below FWHM, or go down until fit R^2 < R_threshold
        n = 0
        R_squared = -np.inf
        while n < 5 or R_squared < 0.9:
            self.focuser.go_to_position(self.focuser.get_position() - self.autofocus_properties['step_width'])
            positions.append( self.focuser.get_position() )
            sleep(0.5)
            img = self.camera.take_image(self.autofocus_properties['gain'], self.autofocus_properties['exp_time'])
            fwhm, fwhm_std_dev = helperfunctions.get_FWHM(img) 
            FWHMs.append( fwhm )
            FWHM_std_devs.append( fwhm_std_dev )

            # remake the fit
            popt, pcov = curve_fit(fun, np.array(positions), FWHMs, p0=p0)

            # update the loop conditions
            n += 1
            R_squared = r2_score(FWHMs, fun(np.array(positions), popt))

            if n > 10:
                # go to the other side of the maximum and measure N_steps_positive points
                self.focuser.go_to_position(np.max(positions))
                sleep(0.5)
                for n in range(self.autofocus_properties['N_steps_positive']):
                    self.focuser.go_to_position(self.focuser.get_position() + self.autofocus_properties['step_width'])
                    positions.append( self.focuser.get_position() )
                sleep(0.5)
                img = self.camera.take_image(self.autofocus_properties['gain'], self.autofocus_properties['exp_time'])
                fwhm, fwhm_std_dev = helperfunctions.get_FWHM(img) 
                FWHMs.append( fwhm )
                FWHM_std_devs.append( fwhm_std_dev )
                self._log_info("Step {}, FWHM={} pm {}".format(n, fwhm, fwhm_std_dev))

                # remake the fit
                popt, pcov = curve_fit(fun, np.array(positions), FWHMs, p0=p0)
                R_squared = r2_score(FWHMs, fun(np.array(positions), popt))
                

                if R_squared > 0.9:
                    break



        position_minimal_FWHM = popt[1]

        return position_minimal_FWHM

