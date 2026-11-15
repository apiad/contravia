"""Where the sun, the moon, the planets and the stars are over Havana at a given moment.

Everything here is a pure function of time. The sun follows NOAA's solar position equations, the moon the
main terms of Meeus's lunar series (Astronomical Algorithms, chapter 47), and the planets JPL's approximate
Keplerian elements (Standish, valid 1800-2050). Each is within a fraction of a degree of a full ephemeris,
far below what a player can see, and needs no data file.

Angles are degrees. A naive datetime is local Havana time, as the game's clock gives it; an aware one is
converted. Altitude is measured from the horizon and azimuth from north toward east. The game's frame has
x east, z south and y up, so north is -z.
"""

import math
from dataclasses import dataclass
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import numpy as np

LAT, LON = 23.136, -82.359
TZ = ZoneInfo("America/Havana")
EARTH_RADIUS_KM = 6378.14


@dataclass(frozen=True)
class Body:
    """A body on the sky: altitude and azimuth, its equatorial place of date, magnitude and lit fraction."""

    alt: float
    az: float
    ra: float
    dec: float
    mag: float = 0.0
    lit: float = 1.0


def _sin(d):
    return np.sin(np.radians(d))


def _cos(d):
    return np.cos(np.radians(d))


def to_utc(t: datetime) -> datetime:
    """The moment t as an aware UTC datetime; a naive t is Havana's local time, summer time included."""
    if t.tzinfo is None:
        t = t.replace(tzinfo=TZ)
    return t.astimezone(UTC)


def julian_day(t: datetime) -> float:
    """The Julian day of the moment t."""
    return to_utc(t).timestamp() / 86400.0 + 2440587.5


def _centuries(t: datetime) -> float:
    return (julian_day(t) - 2451545.0) / 36525.0


def lst(t: datetime) -> float:
    """Local mean sidereal time at Havana, in degrees."""
    d = julian_day(t) - 2451545.0
    T = d / 36525.0
    gmst = 280.46061837 + 360.98564736629 * d + 0.000387933 * T * T
    return float((gmst + LON) % 360.0)


def obliquity(t: datetime) -> float:
    """The tilt of the earth's axis to the ecliptic, in degrees."""
    T = _centuries(t)
    return 23.439291 - 0.0130042 * T


def altaz(ra, dec, t: datetime):
    """Altitude and azimuth of equatorial places of date (scalars or arrays) seen from Havana at t."""
    h = lst(t) - np.asarray(ra, float)
    dec = np.asarray(dec, float)
    sin_alt = _sin(LAT) * _sin(dec) + _cos(LAT) * _cos(dec) * _cos(h)
    alt = np.degrees(np.arcsin(np.clip(sin_alt, -1.0, 1.0)))
    az = np.degrees(np.arctan2(_sin(h), _cos(h) * _sin(LAT) - np.tan(np.radians(dec)) * _cos(LAT))) + 180.0
    return alt, az % 360.0


def direction(alt, az) -> np.ndarray:
    """Unit vectors toward (alt, az) in the game's frame (x east, y up, z south); shape (..., 3)."""
    alt, az = np.asarray(alt, float), np.asarray(az, float)
    return np.stack([_cos(alt) * _sin(az), _sin(alt), -_cos(alt) * _cos(az)], axis=-1)


def equatorial_to_local(t: datetime) -> np.ndarray:
    """The 3x3 rotation taking a unit vector of (RA, Dec) of date into the game's frame at t.

    Columns are the images of the vernal point, of RA 90 on the equator and of the celestial north pole, so
    `R @ v` turns a whole star catalogue with one product.
    """
    alt, az = altaz([0.0, 90.0, 0.0], [0.0, 0.0, 90.0], t)
    return direction(alt, az).T


def unit_vectors(ra, dec) -> np.ndarray:
    """Equatorial unit vectors (x toward RA 0, z toward the north pole) for RA and Dec arrays; shape (N, 3)."""
    ra, dec = np.asarray(ra, float), np.asarray(dec, float)
    return np.stack([_cos(dec) * _cos(ra), _cos(dec) * _sin(ra), _sin(dec)], axis=-1)


def precess(ra, dec, t: datetime):
    """Move J2000 places to the equinox of t (Meeus 21.2); about a third of a degree by 2026."""
    T = _centuries(t)
    zeta = (2306.2181 * T + 0.30188 * T * T) / 3600.0
    z = (2306.2181 * T + 1.09468 * T * T) / 3600.0
    theta = (2004.3109 * T - 0.42665 * T * T) / 3600.0
    ra, dec = np.asarray(ra, float), np.asarray(dec, float)
    A = _cos(dec) * _sin(ra + zeta)
    B = _cos(theta) * _cos(dec) * _cos(ra + zeta) - _sin(theta) * _sin(dec)
    C = _sin(theta) * _cos(dec) * _cos(ra + zeta) + _cos(theta) * _sin(dec)
    return (np.degrees(np.arctan2(A, B)) + z) % 360.0, np.degrees(np.arcsin(np.clip(C, -1.0, 1.0)))


def _ecliptic_to_equatorial(lon, lat, eps):
    ra = np.degrees(np.arctan2(_sin(lon) * _cos(eps) - np.tan(np.radians(lat)) * _sin(eps), _cos(lon)))
    dec = np.degrees(np.arcsin(_sin(lat) * _cos(eps) + _cos(lat) * _sin(eps) * _sin(lon)))
    return float(ra % 360.0), float(dec)


def sun(t: datetime) -> Body:
    """The sun, by NOAA's solar position equations."""
    T = _centuries(t)
    L0 = (280.46646 + T * (36000.76983 + 0.0003032 * T)) % 360.0
    M = 357.52911 + T * (35999.05029 - 0.0001537 * T)
    C = (_sin(M) * (1.914602 - T * (0.004817 + 0.000014 * T)) + _sin(2 * M) * (0.019993 - 0.000101 * T)
         + _sin(3 * M) * 0.000289)
    omega = 125.04 - 1934.136 * T
    lam = L0 + C - 0.00569 - 0.00478 * _sin(omega)
    eps = obliquity(t) + 0.00256 * _cos(omega)
    ra, dec = _ecliptic_to_equatorial(lam, 0.0, eps)
    alt, az = altaz(ra, dec, t)
    return Body(float(alt), float(az), ra, dec, mag=-26.7, lit=1.0)


# Meeus table 47.A and 47.B, the largest terms: (D, M, M', F, coefficient in 1e-6 degrees).
_MOON_LON = ((0, 0, 1, 0, 6288774), (2, 0, -1, 0, 1274027), (2, 0, 0, 0, 658314), (0, 0, 2, 0, 213618),
             (0, 1, 0, 0, -185116), (0, 0, 0, 2, -114332), (2, 0, -2, 0, 58793), (2, -1, -1, 0, 57066),
             (2, 0, 1, 0, 53322), (2, -1, 0, 0, 45758), (0, 1, -1, 0, -40923), (1, 0, 0, 0, -34720),
             (0, 1, 1, 0, -30383))
_MOON_DIST = ((0, 0, 1, 0, -20905355), (2, 0, -1, 0, -3699111), (2, 0, 0, 0, -2955968), (0, 0, 2, 0, -569925))
_MOON_LAT = ((0, 0, 0, 1, 5128122), (0, 0, 1, 1, 280602), (0, 0, 1, -1, 277693), (2, 0, 0, -1, 173237),
             (2, 0, -1, 1, 55413), (2, 0, -1, -1, 46271))


def moon(t: datetime) -> Body:
    """The moon seen from Havana (parallax included), with its lit fraction, by Meeus's main terms."""
    T = _centuries(t)
    Lp = 218.3164477 + 481267.88123421 * T
    D = 297.8501921 + 445267.1114034 * T
    M = 357.5291092 + 35999.0502909 * T
    Mp = 134.9633964 + 477198.8675055 * T
    F = 93.2720950 + 483202.0175233 * T

    def arg(d, m, mp, f):
        return d * D + m * M + mp * Mp + f * F

    lon = Lp + sum(c * _sin(arg(d, m, mp, f)) for d, m, mp, f, c in _MOON_LON) / 1e6
    lat = sum(c * _sin(arg(d, m, mp, f)) for d, m, mp, f, c in _MOON_LAT) / 1e6
    dist = 385000.56 + sum(c * _cos(arg(d, m, mp, f)) for d, m, mp, f, c in _MOON_DIST) / 1000.0
    ra, dec = _ecliptic_to_equatorial(lon, lat, obliquity(t))
    alt, az = altaz(ra, dec, t)
    parallax = math.degrees(math.asin(EARTH_RADIUS_KM / dist))
    alt = float(alt) - parallax * math.cos(math.radians(float(alt)))  # seen from the surface, not the centre
    i = (180.0 - D % 360.0 - 6.289 * _sin(Mp) + 2.100 * _sin(M) - 1.274 * _sin(2 * D - Mp)
         - 0.658 * _sin(2 * D) - 0.214 * _sin(2 * Mp) - 0.110 * _sin(D))
    lit = float((1.0 + _cos(i)) / 2.0)
    return Body(alt, float(az), ra, dec, mag=-12.7 + 2.5 * (1.0 - lit) * 3.0, lit=lit)


# JPL approximate Keplerian elements, table 1 (1800-2050): a, e, I, L, long. perihelion, long. node,
# each with its rate per century.
_PLANETS = {
    "mercury": ((0.38709927, 0.00000037), (0.20563593, 0.00001906), (7.00497902, -0.00594749),
                (252.25032350, 149472.67411175), (77.45779628, 0.16047689), (48.33076593, -0.12534081)),
    "venus": ((0.72333566, 0.00000390), (0.00677672, -0.00004107), (3.39467605, -0.00078890),
              (181.97909950, 58517.81538729), (131.60246718, 0.00268329), (76.67984255, -0.27769418)),
    "earth": ((1.00000261, 0.00000562), (0.01671123, -0.00004392), (-0.00001531, -0.01294668),
              (100.46457166, 35999.37244981), (102.93768193, 0.32327364), (0.0, 0.0)),
    "mars": ((1.52371034, 0.00001847), (0.09339410, 0.00007882), (1.84969142, -0.00813131),
             (-4.55343205, 19140.30268499), (-23.94362959, 0.44441088), (49.55953891, -0.29257343)),
    "jupiter": ((5.20288700, -0.00011607), (0.04838624, -0.00013253), (1.30439695, -0.00183714),
                (34.39644051, 3034.74612775), (14.72847983, 0.21252668), (100.47390909, 0.20469106)),
    "saturn": ((9.53667594, -0.00125060), (0.05386179, -0.00050991), (2.48599187, 0.00193609),
               (49.95424423, 1222.49362201), (92.59887831, -0.41897216), (113.66242448, -0.28867794)),
}
# Absolute magnitude and phase coefficient (per degree), close to the Astronomical Almanac's.
_MAG = {"mercury": (-0.42, 0.038), "venus": (-4.40, 0.009), "mars": (-1.52, 0.016), "jupiter": (-9.40, 0.005),
        "saturn": (-8.88, 0.044)}


def _heliocentric(name: str, T: float) -> np.ndarray:
    """Ecliptic J2000 position of a planet around the sun, in astronomical units."""
    (a, da), (e, de), (I, dI), (L, dL), (w, dw), (O, dO) = _PLANETS[name]
    a, e, I, L, w, O = a + da * T, e + de * T, I + dI * T, L + dL * T, w + dw * T, O + dO * T
    M = math.radians((L - w + 180.0) % 360.0 - 180.0)
    E = M + e * math.sin(M)
    for _ in range(8):  # Kepler's equation by Newton's method
        E -= (E - e * math.sin(E) - M) / (1.0 - e * math.cos(E))
    xp, yp = a * (math.cos(E) - e), a * math.sqrt(1.0 - e * e) * math.sin(E)
    om, Om, i = math.radians(w - O), math.radians(O), math.radians(I)
    co, so, cO, sO, ci, si = math.cos(om), math.sin(om), math.cos(Om), math.sin(Om), math.cos(i), math.sin(i)
    return np.array([(co * cO - so * sO * ci) * xp + (-so * cO - co * sO * ci) * yp,
                     (co * sO + so * cO * ci) * xp + (-so * sO + co * cO * ci) * yp,
                     (so * si) * xp + (co * si) * yp])


def planets(t: datetime) -> dict[str, Body]:
    """Mercury, Venus, Mars, Jupiter and Saturn as seen from Havana, with their magnitudes."""
    T = _centuries(t)
    earth = _heliocentric("earth", T)
    out = {}
    for name, (h, k) in _MAG.items():
        p = _heliocentric(name, T)
        g = p - earth
        r, delta = float(np.linalg.norm(p)), float(np.linalg.norm(g))
        lon = math.degrees(math.atan2(g[1], g[0]))
        lat = math.degrees(math.asin(g[2] / delta))
        ra, dec = _ecliptic_to_equatorial(lon, lat, 23.4392911)  # J2000 ecliptic and equator
        ra, dec = precess(ra, dec, t)
        alt, az = altaz(ra, dec, t)
        phase = math.degrees(math.acos(max(-1.0, min(1.0, (r * r + delta * delta - float(earth @ earth))
                                                     / (2 * r * delta)))))
        bend = k * phase if name != "venus" else 0.0009 * phase + 0.000239 * phase ** 2 - 6.5e-7 * phase ** 3
        out[name] = Body(float(alt), float(az), float(ra), float(dec),
                         mag=h + 5 * math.log10(r * delta) + bend, lit=(1 + math.cos(math.radians(phase))) / 2)
    return out
