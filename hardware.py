from typing import Protocol
from enum import Enum
from astropy.coordinates import Angle, SkyCoord
import threading
from datetime import datetime
from pilot import Pilot, CameraPilot, MountPilot, FocuserPilot, FilterWheelPilot
import numpy as np
import logging
from datetime import datetime, timedelta
import time


class HardwareStatus(Enum):
    UNKNOWN = "Unknown"
    OPERATIONAL = "Operational"
    INOPERATIONAL = "Inoperational"
    INUSE = "In Use"


class Hardware(Protocol):
    hardware_id: str
    hardware_type: str
    status: HardwareStatus

    pilot: Pilot

    def initialize(self):
        pass

# Class that models a rectangular field of view using two angles


class FieldOfView:
    def __init__(self, x: Angle, y: Angle):
        self.x = x
        self.y = y


class Camera(Hardware):

    def __init__(self, pilot: CameraPilot, hardware_id: str, FOV: FieldOfView, hardware_type="Camera"):
        # Log setup
        self._logger = logging.getLogger("hardware.Camera."+hardware_id)
        self._logger.setLevel(logging.DEBUG)
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        # File handler
        fh = logging.FileHandler("hardware.log")
        fh.setLevel(logging.DEBUG)

        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        fh.setFormatter(formatter)

        # Adding handlers
        self._logger.addHandler(ch)
        self._logger.addHandler(fh)

        self._logger.info(
            f"New instance of Camera beeing created with ID: {hardware_id}")

        self._pilot = pilot
        self.hardware_id = hardware_id
        self.hardware_type = hardware_type
        self.status = None
        self._loop_mode = False
        self._default_binning = 1
        self._default_exposure_time = 1.0  # seconds
        self._default_gain = 100
        self.plate_scale = 1.0  # arcsec/pixel

        self.store_images = False  # TODO: Implement image storage

        self._binning = None
        self._exposure_time = None
        self._gain = None

        self.pointing_model = None
        # Represents the angle between the camera's x-axis and the horizon
        self._rotation_angle = None
        self.FOV = None
        self.stop_loop = True

        self._exposure_thread = None
        self._image_list = []

        self.last_image = None

    def __str__(self):
        return "Camera: " + self.hardware_id + "."

    # Loggers with name
    def _log_debug(self, message: str, **kwargs) -> None:
        self._logger.debug(self.hardware_id + ": " + message, **kwargs)

    def _log_info(self, message: str, **kwargs) -> None:
        self._logger.info(self.hardware_id + ": " + message, **kwargs)

    def _log_warning(self, message: str, **kwargs) -> None:
        self._logger.warning(self.hardware_id + ": " + message, **kwargs)

    def _log_error(self, message: str, **kwargs) -> None:
        self._logger.error(self.hardware_id + ": " + message, **kwargs)

    def initialize(self):
        assert self._pilot is not None, "Camera pilot not set."
        if not self._pilot.is_connected():
            try:
                self._pilot.connect()
            except Exception as e:
                self._log_error("Could not connect to camera: " + str(e))
                self._status = HardwareStatus.INOPERATIONAL
                return
        self._status = HardwareStatus.OPERATIONAL
        self._log_info("Initializing camera with default parameters.")
        self._exposure_time = self._default_exposure_time
        self._gain = self._default_gain

    def activate_loop_mode(self):
        self._loop_mode = True
        self._log_info("Loop mode activated.")

    def deactivate_loop_mode(self):
        self._loop_mode = False
        self._log_info("Loop mode deactivated.")

    def start_exposure(self):
        self._log_info("Setting up exposure.")
        self._log_debug("Exposure time: " + str(self._exposure_time) + " s.")
        self._log_debug("Gain: " + str(self._gain) + ".")
        self._log_debug("Binning: " + str(self._binning) + ".")
        self._exposure_thread = threading.Thread(target=self._start_exposure)
        self._exposure_thread.start()
        exposure_start_time = datetime.utcnow()
        self._log_info(f"Exposure started at {exposure_start_time}.")

    def _start_exposure(self):
        """This function is called in a thread to start the exposure."""
        self._pilot.start_exposure(
            self._exposure_time, self._gain, self._binning)
        while not self._pilot.ready:
            time.sleep(0.005)  # TODO: Check time interval
        self._log_debug("Exposure finished.")
        self._log_debug("Getting image.")
        current_image = self._pilot.get_image()
        self._log_debug("Image retrieved.")
        # TODO: Check if this works for a large amount of images (memory)
        self.last_image = current_image
        if self.store_images:
            self._image_list.append(current_image)

        if self._loop_mode:
            self._log_debug("Loop mode is active. Starting next exposure.")
            self._start_exposure()

    def stop_exposure(self):
        self._log_info("Force stopping exposure.")
        self._loop_mode = False
        self._exposure_thread.join()
        self._pilot.stop_exposure()

    def get_images(self):
        """Returns a list of all images taken since the last call of this function."""
        self._log_info("Returning image list.")
        temp_list = self._image_list
        # clearing image list
        self._image_list = []
        return temp_list

    def get_last_image(self):
        """Returns the last image taken."""
        self._log_debug("Returning last image.")
        return self.last_image

    def set_exposure_time(self, exposure_time: float):
        """Exposure time in seconds."""
        self._log_info("Setting exposure time to " +
                       str(exposure_time) + " s.")
        self._exposure_time = float(exposure_time)

    def set_gain(self, gain: int):
        """Gain as integer."""
        self._log_info("Setting gain to " +
                       str(gain) + ".")
        self._gain = int(gain)
    
    def take_image(self, gain, exp_time):
        """Takes a picture with gain and time, returns nump.ndarray"""
        self._pilot.take_image(gain, exp_time)

class Mount(Hardware):

    def __init__(self, pilot: MountPilot, hardware_id: str, hardware_type="Mount"):
        # Log setup
        self._logger = logging.getLogger("hardware.Mount."+hardware_id)
        self._logger.setLevel(logging.DEBUG)
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        # File handler
        fh = logging.FileHandler("hardware.log")
        fh.setLevel(logging.DEBUG)

        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        fh.setFormatter(formatter)

        # Adding handlers
        self._logger.addHandler(ch)
        self._logger.addHandler(fh)

        self._logger.info(
            f"New instance of Mount beeing created with ID: {hardware_id}")

        self._pilot = pilot
        self.hardware_id = hardware_id
        self.hardware_type = hardware_type
        self.status = None

        self.command_delay = 0.5
        self.update_rate = 2.0

    def __str__(self):
        return "Mount: " + self.hardware_id + "."

    # Loggers with name
    def _log_debug(self, message: str, **kwargs) -> None:
        self._logger.debug(self.hardware_id + ": " + message, **kwargs)

    def _log_info(self, message: str, **kwargs) -> None:
        self._logger.info(self.hardware_id + ": " + message, **kwargs)

    def _log_warning(self, message: str, **kwargs) -> None:
        self._logger.warning(self.hardware_id + ": " + message, **kwargs)

    def _log_error(self, message: str, **kwargs) -> None:
        self._logger.error(self.hardware_id + ": " + message, **kwargs)

    def initialize(self):
        assert self._pilot is not None, "Mount pilot not set."
        if not self._pilot.is_connected():
            try:
                self._pilot.connect()
            except Exception as e:
                self._log_error("Could not connect to mount: " + str(e))
                self._status = HardwareStatus.INOPERATIONAL

        self._status = HardwareStatus.OPERATIONAL
        self._log_info("Initializing mount.")
        self._pilot.initialize()

    def slew_to(self, target: SkyCoord):
        self._log_info("Slewing to target.")
        try:
            self._pilot.slew_to(target)
        except Exception as e:
            self._log_error("Could not slew to target: " + str(e))

    def get_angles(self):
        """Returns the current Right-Ascension and Declination angles of the mount in degrees."""
        self._log_info("Getting current angles of the mount.")
        return self._pilot.get_angles()

    def get_status(self):
        """Returns the current status of the mount."""
        self._log_debug("Getting status.")
        return self._status

    def set_slew_rate(self, ra_rate: float, dec_rate: float):
        """Rates expected in arcsec/s."""
        self._log_info(
            f"Trying to set slew rate to RA_dot: {ra_rate} arcsec/s, DEC_dot: {dec_rate} arcsec/s.")
        try:
            self._pilot.set_slew_rate(ra_rate, dec_rate)
        except Exception as e:
            self._log_error("Erronous slew rate: " + str(e))

    def is_slewing(self):
        return self._pilot.is_slewing()

    def stop_slew(self):
        self._log_info("Stopping slew.")
        self._pilot.stop_slew()


class Focuser(Hardware):
    def __init__(self, focuser_pilot: FocuserPilot, hardware_id: str):
        self._logger = logging.getLogger("hardware.Focuser." + hardware_id)
        self._logger.setLevel(logging.DEBUG)
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        # File handler
        fh = logging.FileHandler("hardware.log")
        fh.setLevel(logging.DEBUG)

        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        fh.setFormatter(formatter)

        # Adding handlers
        self._logger.addHandler(ch)
        self._logger.addHandler(fh)

        self._logger.info(
              f"New instance of Focuser beeing created with ID: {hardware_id}")

        self.hardware_type = "Focuser"
        self.hardware_id = hardware_id

        self._pilot = focuser_pilot
    
        # place to save some positions (e.g. for each filter in a filter wheel)
        self._saved_positions = {}
        self._saved_positions['initial_position'] = self.get_position()

     # Loggers with name
    def _log_debug(self, message: str, **kwargs) -> None:
        self._logger.debug(self.hardware_id + ": " + message, **kwargs)

    def _log_info(self, message: str, **kwargs) -> None:
        self._logger.info(self.hardware_id + ": " + message, **kwargs)

    def _log_warning(self, message: str, **kwargs) -> None:
        self._logger.warning(self.hardware_id + ": " + message, **kwargs)

    def _log_error(self, message: str, **kwargs) -> None:
        self._logger.error(self.hardware_id + ": " + message, **kwargs)

    def go_to_position(self, position):
        self._log_info(f"Moving focus to {position}.")
        self._pilot.go_to_position(position)
    
    def get_position(self):
        p = self._pilot.get_position()
        self._log_info(f'Request position: {p}')
        return p
    
    def save_current_position(self, name):
        self._log_info(f'Saving position {name} as {self.get_position()}')
        self._saved_positions[name] = self.get_position()

class FilterWheel(Hardware):
    def __init__(self, filterwheel_pilot: FilterWheelPilot, hardware_id: str):
        self._logger = logging.getLogger("hardware.FilterWheel." + hardware_id)
        self._logger.setLevel(logging.DEBUG)
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        # File handler
        fh = logging.FileHandler("hardware.log")
        fh.setLevel(logging.DEBUG)

        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        fh.setFormatter(formatter)

        # Adding handlers
        self._logger.addHandler(ch)
        self._logger.addHandler(fh)

        self._logger.info(
              f"New instance of FilterWheel beeing created with ID: {hardware_id}")

        self.hardware_type = "FilterWheel"
        self.hardware_id = hardware_id

        self._pilot = filterwheel_pilot
    
     # Loggers with name
    def _log_debug(self, message: str, **kwargs) -> None:
        self._logger.debug(self.hardware_id + ": " + message, **kwargs)

    def _log_info(self, message: str, **kwargs) -> None:
        self._logger.info(self.hardware_id + ": " + message, **kwargs)

    def _log_warning(self, message: str, **kwargs) -> None:
        self._logger.warning(self.hardware_id + ": " + message, **kwargs)

    def _log_error(self, message: str, **kwargs) -> None:
        self._logger.error(self.hardware_id + ": " + message, **kwargs)
    
    def get_names(self):
        return self._pilot.get_names()

    def select_filter(self, filter):
        self._log_info(f"Selecting filter {filter}.")
        self._pilot.select_filter(filter)
    
    def get_current_filter(self):
        p = self._pilot.get_current_filter()
        self._log_info(f'Request position: {p}')
        return p