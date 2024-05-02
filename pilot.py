from typing import Protocol
from astropy import units
from astropy.coordinates import SkyCoord
import numpy as np
from alpaca.telescope import Telescope, TelescopeAxes
from alpaca.focuser import Focuser
from alpaca.filterwheel import FilterWheel
import time
import logging


class Pilot(Protocol):
    def connect(self):
        pass

    def disconnect(self):
        pass

    def is_connected(self):
        pass


class CameraPilot(Pilot):
    def __init__(self):
        self._is_connected = False

    def connect(self):
        self._is_connected = True

    def disconnect(self):
        self._is_connected = False

    def is_connected(self):
        return self._is_connected

    def start_exposure(self, exposure_time, gain, binning):
        pass

    def stop_exposure(self):
        pass

    def get_image(self) -> np.ndarray:
        return np.zeros((101, 101))

    def ready(self):
        return False


class MountPilot(Pilot):
    def __init__(self):
        self._is_connected = False

    def connect(self):
        self._is_connected = True

    def disconnect(self):
        self._is_connected = False

    def is_connected(self):
        return self._is_connected

    def slew_to(self, position: SkyCoord):
        pass

    def get_position(self) -> SkyCoord:
        return SkyCoord(0, 0, unit=units.deg)

    def set_slew_rate(self, ra_rate, dec_rate):
        pass

    def is_slewing(self):
        return False


class DomePilot(Pilot):
    pass


class FocuserPilot(Pilot):
    def __init__(self):
        self._is_connected = False

    def connect(self):
        self._is_connected = True

    def disconnect(self):
        self._is_connected = False

    def is_connected(self):
        return self._is_connected

    def go_to_position(self, positoin):
        pass

    def get_position(self):
        pass

class DummyCameraPilot(CameraPilot):
    def __init__(self):
        self._is_connected = False
        self._logger = logging.getLogger("pilot.DummyCameraPilot")
        self._logger.info("New instance of DummyCameraPilot beeing created")

    def connect(self):
        self._is_connected = True
        self._logger.info("DummyCameraPilot connected")

    def disconnect(self):
        self._is_connected = False
        self._logger.info("DummyCameraPilot disconnected")

    def is_connected(self):
        return self._is_connected

    def start_exposure(self, exposure_time, gain, binning):
        self._logger.info("DummyCameraPilot started exposure")
        self._ready = False
        time.sleep(exposure_time)
        self._ready = True
        self._logger.info("DummyCameraPilot finished exposure")

    def stop_exposure(self):
        self._logger.info("DummyCameraPilot force stopped exposure")

    def get_image(self) -> np.ndarray:
        self._logger.info("DummyCameraPilot returning image")
        return np.zeros((100, 100))

    def ready(self):
        return self._ready
    
class ASCOMASICameraPilot(CameraPilot):
    def __init__(self, adress: str, id: int):
        self._is_connected = False
        self._logger = logging.getLogger("pilot.DummyCameraPilot")
        self._logger.info("New instance of DummyCameraPilot beeing created")
        self.camera = Camera(adress, id)

    def connect(self):
        self._is_connected = True
        self._logger.info("ASCOMASICameraPilot connected")

    def disconnect(self):
        self._is_connected = False
        self._logger.info("ASCOMASICameraPilot disconnected")

    def is_connected(self):
        return self.camera.Connected

    def set_binning(self):
        pass #TODO: implement binning ASCOM

    def start_exposure(self, exposure_time, gain, binning, Light=True):
        self._logger.info("DummyCameraPilot started exposure")
        self._ready = False
        self.camera.gain = gain
        self.camera.set_binning(binning)
        self.camera.StartExposure(exposure_time, Light=Light)
        while not self.camera.ImageReady:
            time.sleep(0.001)
        self._ready = True
        self._logger.info("DummyCameraPilot finished exposure")

    def stop_exposure(self):
        self._logger.info("DummyCameraPilot force stopped exposure")

    def get_image(self) -> np.ndarray:
        self._logger.info("DummyCameraPilot returning image")
        return np.array(self.camera.ImageArray).T

    def ready(self):
        return self._ready

    def take_image(self, gain, exp_time):
        self.camera.gain = gain
        self.camera.StartExposure(exp_time, Light=True)
        while not self.camera.ImageReady:
            time.sleep(0.1)
        img_array = self.camera.ImageArray
        return np.array(img_array).T

class DummyMountPilot(MountPilot):
    def __init__(self):
        self._is_connected = False
        self._logger = logging.getLogger("pilot.DummyMountPilot")
        self._logger.info("New instance of DummyMountPilot beeing created")
        self._is_slewing = False

    def initialize(self):
        self._logger.info("Initializing")

    def connect(self):
        self._is_connected = True
        self._logger.info("DummyMountPilot connected")

    def disconnect(self):
        self._is_connected = False
        self._logger.info("DummyMountPilot disconnected")

    def is_connected(self):
        return self._is_connected

    def slew_to(self, position: SkyCoord):
        self._logger.info("DummyMountPilot slewing to " + str(position))
        self._is_slewing = True
        time.sleep(0.5)
        self._is_slewing = False

    def get_position(self):
        return SkyCoord(0, 0, unit=units.deg)

    def set_slew_rate(self, ra_rate, dec_rate):
        self._logger.info(
            f"DummyMountPilot setting slew rate to  {ra_rate} arcsec/s, DEC_dot: {dec_rate} arcsec/s.")

    def stop_slew(self):
        self._logger.info("DummyMountPilot stopping slew")

    def is_slewing(self):
        return self._is_slewing


class ASCOMMountPilot(MountPilot):
    def __init__(self, mount_adress: str, mount_number=0):
        self._is_connected = False
        self._logger = logging.getLogger("pilot.ASCOMMountPilot")
        self._logger.info("New instance of ASCOMMountPilot beeing created")
        self.telescope = Telescope(mount_adress, mount_number)
        self._logger.info("ASCOMMountPilot created and connected.")
        self._is_slewing = False

    def initialize(self):
        self.telescope.FindHome()

    def connect(self):
        self._is_connected = True
        self._logger.info("DummyMountPilot connected")

    def is_connected(self):
        return self.telescope.Connected

    def slew_to(self, position: SkyCoord):
        self._logger.info("ASCOMMountPilot slewing to " + str(position))
        self.telescope.SlewToCoordinatesAsync(
            float(position.ra.hourangle), float(position.dec.degree))

    def get_position(self):
        return SkyCoord(ra=self.telescope.RightAscension*u.hourangle, dec=self.telescope.Delination*u.degree, frame='icrs')

    def set_slew_rate(self, ra_rate, dec_rate):
        """Set the slew rate of the mount in deg/s as input."""
        self.telescope.MoveAxis(TelescopeAxes(
            0), ra_rate)
        self.telescope.MoveAxis(TelescopeAxes(1), dec_rate)

        self._logger.info(
            f"ASCOMMountPilot setting slew rate to  {ra_rate} arcsec/s, DEC_dot: {dec_rate} arcsec/s.")

    def stop_slew(self):
        self._logger.info("Stopping slew")
        self.telescope.AbortSlew()

    def is_slewing(self):
        return self.telescope.Slewing


class ASCOMEQMountPilot(MountPilot):
    def __init__(self, mount_adress: str, mount_number=0):
        self._is_connected = False
        self._logger = logging.getLogger("pilot.ASCOMMountPilot")
        self._logger.info("New instance of ASCOMMountPilot beeing created")
        self.telescope = Telescope(mount_adress, mount_number)
        self._logger.info("ASCOMMountPilot created and connected.")
        self._is_slewing = False

    def initialize(self):
        pass
        self.stop_slew()
        # self.telescope.FindHome()

    def connect(self):
        self._is_connected = True
        self._logger.info("DummyMountPilot connected")

    def is_connected(self):
        return self.telescope.Connected

    def slew_to(self, position: SkyCoord):
        self._logger.info("ASCOMMountPilot slewing to " + str(position))
        self.telescope.SlewToCoordinatesAsync(
            float(position.ra.hourangle), float(position.dec.degree))

    def get_position(self):
        return SkyCoord(ra=self.telescope.RightAscension*u.hourangle, dec=self.telescope.Delination*u.degree, frame='icrs')

    def set_slew_rate(self, ra_rate, dec_rate):
        """Set the slew rate of the mount in deg/s as input."""
        self.telescope.MoveAxis(TelescopeAxes(
            0), -ra_rate)
        self.telescope.MoveAxis(TelescopeAxes(1), -dec_rate)

        self._logger.info(
            f"ASCOMMountPilot setting slew rate to  {ra_rate} deg/s, DEC_dot: {dec_rate} deg/s.")

    def stop_slew(self):
        self._logger.info("Stopping slew")
        self.telescope.MoveAxis(TelescopeAxes(0), 0)
        self.telescope.MoveAxis(TelescopeAxes(1), 0)
        self.telescope.AbortSlew()

    def is_slewing(self):
        return self.telescope.Slewing


class DummyFocuserPilot(FocuserPilot):
    def __init__(self):
        self._is_connected = False
        self._logger = logging.getLogger("pilot.DummyFocuserPilot")
        self._logger.info("New instance of DummyFocuserPilot beeing created")

    def connect(self):
        self._is_connected = True
        self._logger.info("DummyFocuserPilot connected")

    def disconnect(self):
        self._is_connected = False
        self._logger.info("DummyFocuserPilot disconnected")

    def is_connected(self):
        return self._is_connected

    def get_position(self):
        return 0

    def do_step(self, direction, step):
        self._logger.info("DummyFocuserPilot doing step " +
                          str(step) + " in direction " + str(direction))

class ASCOMFocuserPilot(FocuserPilot):
    def __init__(self, focuser_adress: str, focuser_number=0):
        self._is_connected = False
        self._logger = logging.getLogger("pilot.ASCOMFocuserPilot")
        self._logger.info("New instance of ASCOMFocuserPilot beeing created")
        self._focuser = Focuser(focuser_adress, focuser_number)
        self._logger.info("ASCOMFocuserPilot created and connected.")
        self.is_moving = False

    def connect(self):
        self._is_connected = True
        self._logger.info("ASCOMFocuserPilot connected")

    def disconnect(self):
        self._is_connected = False
        self._logger.info("ASCOMFocuserPilot disconnected")

    def is_connected(self):
        return self._is_connected

    def get_position(self):
        return self._focuser.Position

    def go_to_position(self, position):
        self._logger.info(f"ASCOMFocuserPilot doing moving to {position}")
        self.is_moving = True
        self._focuser.Move(position)
        while self._focuser.IsMoving:
            time.sleep(0.005)
        self.is_moving = False

class FilterWheelPilot(Pilot):
    def __init__(self):
        self._is_connected = False

    def connect(self):
        self._is_connected = True

    def disconnect(self):
        self._is_connected = False

    def is_connected(self):
        return self._is_connected

    def select_filter(self, filter):
        pass

    def get_current_filter(self):
        pass

    def get_names(self):
        pass

class ASCOMFilterWheelPilot(FilterWheelPilot):
    def __init__(self, focuser_adress: str, focuser_number=0):
        self._is_connected = False
        self._logger = logging.getLogger("pilot.ASCOMFilterWheelPilot")
        self._logger.info("New instance of ASCOMFilterWheelPilot beeing created")
        self._filterwheel = FilterWheel(focuser_adress, focuser_number)
        self._logger.info("ASCOMFilterWheelPilot created and connected.")
        self.is_moving = False
        self.name_to_position = {"L": 0, "g'": 1, "r'": 2, "i'": 3, "z-s'": 4, "Ha": 5, "OIII": 6, "SII": 7}

    def connect(self):
        self._is_connected = True
        self._logger.info("ASCOMFilterWheelPilot connected")

    def disconnect(self):
        self._is_connected = False
        self._logger.info("ASCOMFilterWheelPilot disconnected")

    def is_connected(self):
        return self._is_connected
    
    def get_names(self):
        return list(self.name_to_position.keys())

    def get_current_filter(self):
        return list(self.name_to_position.keys())[list(self.name_to_position.values()).index(self._filterwheel.Position)]

    def select_filter(self, filter_name):
        self._logger.info(f"ASCOMFilterWheelPilot doing moving to {filter_name} with position number {self.name_to_position[filter_name]}")
        self.is_moving = True
        self._filterwheel.Position = self.name_to_position[filter_name]
        while self._filterwheel.Position == -1:
            time.sleep(0.01)
        self.is_moving = False