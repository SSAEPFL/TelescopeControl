from pickletools import read_decimalnl_short
from astropy.coordinates import SkyCoord, EarthLocation
from astropy import units as u

from scipy.optimize import minimize
from traitlets import DottedObjectName
from datetime import datetime

import numpy as np


def loss(parameters, obs_list, com_list):
    """Parameters expected in arcseconds, obs and com are SkyCoord lists"""
    err_ra = 0
    err_dec = 0
    for idx, obs in enumerate(obs_list):
        est = model(com_list[idx], *parameters)
        err_ra += np.abs(est.ra.arcsecond - obs.ra.arcsecond)
        err_dec += np.abs(est.dec.arcsecond - obs.dec.arcsecond)

    return np.sqrt((err_ra + err_dec)**2)


def get_corrections(angle, IH, ID, NP, CH, MA, ME):
    """Parameters expected in arcseconds"""
    dra = (IH + CH*1/np.cos(angle.dec.radian) + NP*np.tan(angle.dec.radian) - MA*np.cos(
        angle.ra.radian) * np.tan(angle.dec.radian) + ME*np.sin(angle.ra.radian) * np.tan(angle.dec.radian))*u.arcsecond

    ddec = (ID + MA*np.sin(angle.ra.radian) + ME *
            np.cos(angle.ra.radian))*u.arcsecond

    return dra, ddec


def model(angle, IH, ID, NP, CH, MA, ME):
    dra, ddec = get_corrections(angle, IH, ID, NP, CH, MA, ME)
    return SkyCoord(ra=angle.ra+dra, dec=angle.dec+ddec, unit=u.arcsecond)


def inverse_model(angle, IH, ID, NP, CH, MA, ME):
    dra, ddec = get_corrections(angle, IH, ID, NP, CH, MA, ME)
    return SkyCoord(ra=angle.ra-dra, dec=angle.dec-ddec, unit=u.arcsecond)


def estimate_alignment(obs_list, com_list):
    """Returns the best fit parameters for the alignment"""
    # Initial guess
    x0 = np.array([0, 0, 0, 0, 0, 0])

    res = minimize(loss, x0, args=(obs_list, com_list))

    return res.x


def get_alignment_coordinates():
    obs_time = datetime.now()
    station_location = EarthLocation(
        lat=46.51650, lon=6.56146, height=397.0)  # EPFL
    alt = [40, 60, 40, 80, 40, 60, 40, 80]
    az = [0, 45, 90, 135, 180, 225, 270, 315]
    com_list = []

    for i in range(len(alt)):

        altaz = SkyCoord(alt=alt[i]*u.deg, az=az[i]*u.deg, frame='altaz',
                         obstime=obs_time, location=station_location)

        radec = altaz.transform_to('icrs')

        com_list.append(radec)

    return com_list


def generate_measurements(n_mes=10, params=[1000, 0, 3600, 0, 3600, 4000]):
    obs_time = datetime.now()
    station_location = EarthLocation(
        lat=46.51650, lon=6.56146, height=397.0)  # EPFL

    params = [1000, 0, 3600, 0, 3600, 4000]

    n_mes = 10
    obs_list = []
    com_list = []
    for i in range(n_mes):
        # generate random ra and dec angles in the sky by generating altitude and azimuth angles at given position
        alt = np.random.uniform(40, 90)
        az = np.random.uniform(0, 360)

        altaz = SkyCoord(alt=alt*u.deg, az=az*u.deg, frame='altaz',
                         obstime=obs_time, location=station_location)

        radec = altaz.transform_to('icrs')

        com_list.append(radec)
        obs = model(radec, *params)
        obs_noisy = SkyCoord(ra=obs.ra.to(u.arcsecond).value+np.random.normal(0, 20),
                             dec=obs.dec.to(u.arcsecond).value+np.random.normal(0, 20), unit=u.arcsecond)
        obs_list.append(obs_noisy)

    return com_list, obs_list


if __name__ == "__main__":

    obs_time = datetime(2021, 5, 1, 12, 0, 0)
    station_location = EarthLocation(lat=52.519, lon=13.408, height=0)

    params = [1000, 0, 3600, 0, 3600, 4000]

    n_mes = 10

    com_list, obs_list = generate_measurements(n_mes, params)

    res = estimate_alignment(obs_list, com_list)
    print(res)

    print("RMSE on calibration: {:.4f} arcseconds".format(
        loss(res, obs_list, com_list)/n_mes))

    com_list_test, obs_list_test = generate_measurements(10, params)

    print("RMSE on test: {:.4f} arcseconds".format(
        loss(res, obs_list_test, com_list_test)/n_mes))
