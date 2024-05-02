from astropy import units as u
import json
import os
import time
from astropy import units, constants
from skyfield import almanac
from skyfield.api import EarthSatellite, load, wgs84
from datetime import datetime
from datetime import timedelta
import dateutil.parser
from calendar import monthrange
from CelestrakTLEDL import download_tle
import numpy as np
from hardware import Camera, FieldOfView
import sep

def load_sat_list(file_name):
    f = open(file_name)

    data = json.load(f)

    ts = load.timescale()

    sat_list = []

    for sat_data in data:
        sat_list.append(EarthSatellite(
            sat_data["tle_1"], sat_data["tle_2"], sat_data["satellite_name"], ts))

    return sat_list


def get_visible_sats(sat_list, station, t):
    visible_sats = []
    for sat in sat_list:
        diff = sat-station
        topo = diff.at(t)
        alt, az, distance = topo.altaz()
        eph = load("de440s.bsp")
        if alt.degrees > 10 and sat.at(t).is_sunlit(eph):
            visible_sats.append(sat)
    return visible_sats


def get_sat_altaz_position(sat, station, t):
    diff = sat-station
    topo = diff.at(t)
    alt, az, distance = topo.altaz()
    # ra, dec, dis = topo.radec('date')
    return alt, az

# function that returns the radec position of a satellite


def get_sat_radec_position(sat, station, t):
    diff = sat-station
    topo = diff.at(t)
    ra, dec, dis = topo.radec('date')
    return ra, dec

# function that returns the altaz velocity of a satellite


def get_sat_altaz_velocity(sat, station, t):

    dt = 0.05
    diff = sat-station
    topo1 = diff.at(t)
    topo2 = diff.at(t-dt/(24*60*60))

    alt1, az1, distance1 = topo1.altaz()
    alt2, az2, distance2 = topo2.altaz()
    alt_dot = (alt1.degrees-alt2.degrees)/dt
    az_dot = (az1.degrees-az2.degrees)/dt
    return alt_dot, az_dot

# function that returns the radec velocity of a satellite


def get_sat_radec_velocity(sat, station, t):

    dt = 0.05
    diff = sat-station
    topo1 = diff.at(t)
    topo2 = diff.at(t-dt/(24*60*60))

    ra1, dec1, dis1 = topo1.radec('date')
    ra2, dec2, dis2 = topo2.radec('date')
    ra_dot = (ra1._degrees-ra2._degrees)/dt
    dec_dot = (dec1.degrees-dec2.degrees)/dt
    return ra_dot, dec_dot


def compute_sunrise_sunset(station, year=2023, month=1, day=1):
    t0 = ts.utc(year, month, day, 0)
    # t1 = t0 plus one day
    t1 = ts.utc(t0.utc_datetime() + timedelta(days=1))
    t, y = almanac.find_discrete(
        t0, t1, almanac.sunrise_sunset(eph, station))
    sunrise = None
    for time, is_sunrise in zip(t, y):
        if is_sunrise:
            sunrise = dateutil.parser.parse(time.utc_iso())
        else:
            sunset = dateutil.parser.parse(time.utc_iso())
    return sunrise, sunset


def is_at_night(tsf, station):
    t = tsf.to_astropy().ymdhms
    sr, ss = compute_sunrise_sunset(
        station, t["year"], t["month"], t["day"])
    if tsf.to_astropy() < sr-timedelta(hours=0.5) or tsf.to_astropy() > ss+timedelta(hours=0.5):
        return True
    else:
        return False


def get_rev_per_day(sat):
    return sat.model.no_kozai/3.14*24*60


def get_orbit_type(sat):
    rev_per_day = get_rev_per_day(sat)
    if rev_per_day < 1.05 and rev_per_day >= 0.95:
        return "GEO"
    if rev_per_day >= 1.05 and rev_per_day < 11:
        return "MEO"
    if rev_per_day >= 11:
        return "LEO"
    else:
        return "HEO"


def live_sat_info(sat, station):
    otype = get_orbit_type(sat)
    while True:
        t = ts.now()
        alt, az = get_sat_altaz_position(sat, station, t)
        v_alt, v_az = get_sat_altaz_velocity(sat, station, t)
        ra, dec = get_sat_radec_position(sat, station, t)
        v_ra, v_dec = get_sat_radec_velocity(sat, station, t)

        if sat.at(t).is_sunlit(eph):
            lit_str = "Object is sunlit."
        else:
            lit_str = "Object is in Earth's shadow."

        os.system("clear")
        print(sat.model.intldesg + " at UTC:", str(t.utc_datetime()))
        print("Orbit Type: " + otype)
        print(
            "Position in the sky: Alt: {}, Az: {}".format(alt, az))
        print(
            "Velocity in the sky: Alt: {:.4f} deg/s, Az: {:.4f} deg/s".format(v_alt, v_az))
        print("Position in the sky: RA: {}, Dec: {}".format(ra, dec))
        print(
            "Velocity in the sky: RA: {:.4f} deg/s, Dec: {:.4f} deg/s".format(v_ra, v_dec))
        print(lit_str)
        time.sleep(0.02)


def find_sat(sat_list, sat_number):

    #for sat in sat_list:
    #    if sat.model.satnum == int(sat_number):
    #       return sat

    for sat in sat_list:
        if sat.model.intldesg == sat_number:
            return sat


def find_passes(sat, station, t0, t1):
    times, events = sat.find_events(
        station, t0, t1, altitude_degrees=10.0)

    print(events)

    if len(events) != 0:
        if events[-1] != 2:
            events = events[:-2]
            times = times[:-2]
        if events[-1] != 2:
            events = events[:-2]
            times = times[:-2]
        if events[0] != 0:
            events = events[1:]
            times = times[1:]
        if events[0] != 0:
            events = events[1:]
            times = times[1:]

    event_names = 'Rise', 'Culm.', 'Set'
    observable_events = []
    observable_times = []

    for i in range(int(len(events)/3)):
        if is_at_night(times[3*i+1], station) and sat.at(times[3*i+1]).is_sunlit(eph):
            observable_events.append(events[3*i])
            observable_events.append(events[3*i+1])
            observable_events.append(events[3*i+2])
            observable_times.append(times[3*i])
            observable_times.append(times[3*i+1])
            observable_times.append(times[3*i+2])

    print("Next passes for Object: "+sat.model.intldesg+"\n")

    for t, ev in zip(observable_times, observable_events):
        alt, az = get_sat_altaz_position(sat, station, t)
        print(str(t.utc_strftime()) + ": " +
              event_names[int(ev)] + "\t at: Alt: {}, Az: {}".format(alt, az))
        if ev == 2:
            print("\n")


def seconds_to_hours(seconds):
    hours = seconds/3600
    return hours


def is_in_fov(sat, station_position, cam, t):
    """In its current version this function just checks if the satellite is inside a cercle of radius defined by the smallest axis of the camera, to get around rotation issues."""
    # TODO: Implement a proper check for the fov
    return True


def rotation_matrix(rotation_angle):
    """This functions expects an angle in degrees and returns a rotation matrix in 2D to derotate the image"""
    return np.array([[np.cos(np.deg2rad(rotation_angle)), np.sin(np.deg2rad(rotation_angle))],
                     [-np.sin(np.deg2rad(rotation_angle)), np.cos(np.deg2rad(rotation_angle))]])

def get_FWHM(image):
    image /= np.mean(image)

    bkg = sep.Background(image)
    sources = sep.extract(image-bkg.back(), 1.5, err=bkg.globalrms)

    fwhm_list = sources['fwhm']

    mean_fwhm = np.mean(fwhm_list)
    std_fwhm = np.std(fwhm_list)
    return mean_fwhm, std_fwhm


if __name__ == "__main__":

    sat_list = load_sat_list("celes_tle.json")
    eph = load("de440s.bsp")

    epfl_station = wgs84.latlon(46.51650, 6.56146, elevation_m=397.0)

    selected_sat = find_sat(sat_list, 25544)

    ts = load.timescale()
    # print(selected_sat.name)

    if selected_sat == None:
        print("Satellite not found")
        exit()

    ts = load.timescale()
    t0 = ts.now()
    t1 = t0+timedelta(days=4)

    visible_sat_list = get_visible_sats(
        sat_list, epfl_station, ts.now()+timedelta(hours=5))

    print(len(visible_sat_list))
    selected_sat = visible_sat_list[102]

    find_passes(selected_sat, epfl_station, t0, t1)

    print(get_orbit_type(selected_sat))

    # live_sat_info(selected_sat, epfl_station)
