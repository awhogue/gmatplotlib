#!/usr/bin/python

import random, sys, datetime, time, os, math
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np

"""
Libraries for working with matplotlib and Google Maps.

Example usage:

import gmatplotlib as gm
# Points should look like [[(40.7, -72.0), (41.3, -72.2), ...],
                           [(37.78, -122.40), (37.8, -122.39)]]
points = ...
gm.plot_points_on_map(points, output_filename='myfig.png')
"""

# Pixels per degree of longitude for each zoom level.  Constant at all longitudes because
# Google Maps uses the Mercator projection.
# Tweaked from https://groups.google.com/d/msg/google-maps-api/b_g0MSc8ats/tLCI7BqdX94J,
# This is in reverse order (zoom level 18 down to 0) because we want to iterate through it in
# that order and choose the tightest zoom that covers our desired area within 640x640 pixels.
_ZOOM_PIXELS = [
  745654.0,                 # zoom level 19
  372827.0,                 # zoom level 18
  186413.5,                 # zoom level 17
   93206.75555555556,       # zoom level 16
   46603.37777777778,       # zoom level 15
   23301.68888888889,       # zoom level 14
   11650.844444444445,      # zoom level 13
    5825.422222222222,      # zoom level 12
    2912.711111111111,      # zoom level 11
    1456.3555555555556,     # zoom level 10
     728.1777777777778,     # zoom level 9
     364.0888888888889,     # zoom level 8
     182.04444444444445,    # zoom level 7
      91.02222222222223,    # zoom level 6
      45.51111111111111,    # zoom level 5
      22.755555555555556,   # zoom level 4
      11.377777777777778,   # zoom level 3
       5.688888888888889,   # zoom level 2
       2.8444444444444446,  # zoom level 1
       1.4222222222222223,  # zoom level 0
]

def compute_zoom_and_size(center, span):
  """Given an area of the map, compute the right zoom level and size, in pixels, to cover it.

  Args:
    center: A list of 2 elements containing the latitude and longitude of the center of the map.
    span: A list of 2 elements containing the span of the map, in degrees latitude and longitude.

  Returns: A pair of zoom level (integer) and size (string) appropriate for passing to
           the Google Static Maps API.
  """
  # Iterate through each zoom level and find the first one where we can fit the map
  # in under 640x640 pixels (the maximum for non-paid use of the static maps API).
  for ii, pixels_per_degree in enumerate(_ZOOM_PIXELS):
    zoom = len(_ZOOM_PIXELS) - ii - 1
    lat_pixels = int(pixels_per_degree * span[0] / math.cos(center[0] * math.pi / 180.0))
    lng_pixels = int(pixels_per_degree * span[1])
    if (lat_pixels <= 640 and lng_pixels <= 640):
      print '  Chose zoom %d, image size %dx%d' % (zoom, lng_pixels, lat_pixels)
      return (zoom, '%dx%d' % (lng_pixels, lat_pixels))

  print 'Error: could not find large enough zoom to cover span (%f,%f)' % (span[0], span[1])
  return (0, '640x640')

# Given a center and span in terms of lat/long, get an appropriate image
# from the google static maps api that covers that area.  Stores the image
# on disk and returns the filename.
def get_map_url(center, zoom, size):
  """Get the URL for a Google Static Map covering the given map area.

  Args:
    center: A list of 2 elements containing the latitude and longitude of the center of the map.
    zoom: The zoom level for the map.
    size: The size, in pixels, of the map.  E.g. '640x640'.

  Returns: A string URL for a map covering the given area.
  """
  return 'http://maps.googleapis.com/maps/api/staticmap?sensor=false' + \
      '&size=%s&center=%f,%f&zoom=%d' % (size, center[0], center[1], zoom)

def get_map_file(center, span, output_dir):
  """Fetch a map image for the given bounds and store it in output_dir.

  Args: 
    center: A list of 2 elements containing the latitude and longitude of the center of the map.
    span: A list of 2 elements containing the span of the map, in degrees latitude and longitude.

  Returns: A string filename pointing to the map image file.
  """
  (zoom, size) = compute_zoom_and_size(center, span)
  url = get_map_url(center, zoom, size)
  (north, south, east, west) = bounds(center, span)
  filename = '%s/%f_%f_%f_%f_%d.png' % (output_dir, north, south, east, west, zoom)
  os.popen('mkdir -p %s' % output_dir)
  os.popen('wget -q -O %s "%s"' % (filename, url))
  return filename


def bounds(center, span):
  """Get the bounds (north, south, east, west) of the map.
  Args:
    center: A list of 2 elements containing the latitude and longitude of the center of the map.
    span: A list of 2 elements containing the span of the map, in degrees latitude and longitude.

  Returns: A 4-tuple containing the bounds (north, south, east, west).
  """
  return (center[0] + span[0],
          center[0] - span[0],
          center[1] + span[1],
          center[1] - span[1])

# Return the p'th percentile value of l.
def ptile(l, p): return l[int(len(l)*p)]

# Filter outlier values outside of the p'th percentile of l.
# E.g. if p=.8, drops values below the 10th and above 90th percentile.
def filter_percentile(l, p):
  s = sorted(l)
  lower = ptile(s, (1.0 - p)/2.0)
  upper = ptile(s, 1.0 - (1.0 - p)/2.0)
  return [x for x in l if (x >= lower and x <= upper)]

def fit_center_span(latitudes, longitudes, percentile=0.7):
  """Given a set of points, compute a reasonable set of map bounds to plot them.

  We take the 70th percentile of the points that are closest to the centroid, then pad
  the span a bit to give some context.

  Args:
    latitudes: A list of latitudes.
    longitudes: A list of longitudes.
    percentile: The percentage of points closest to the centroid to keep.

  Returns: A 2-tuple of the center and span for the bounds in terms of lat/long degrees.
  """
  lats = filter_percentile(latitudes, percentile)
  lngs = filter_percentile(longitudes, percentile)
  center = [np.mean(lats), np.mean(lngs)]

  # Buffer the span by a little bit to give the image some margins.
  min_lat_span = 1.2 * (np.max(lats) - np.min(lats))
  min_lng_span = 1.2 * (np.max(lngs) - np.min(lngs))

  # We want a roughly square patch on the map, so it doesn't end up stretched in weird ways
  # when we plot it.  Since latitude is distored as you move towards the poles in mercator,
  # "square" in terms of degrees is actually a rectangle when rendered.  We stretch out
  # longitude by sec(latitude) so the image comes out square.
  lat_span = max(min_lat_span, min_lng_span)
  lng_span = lat_span * 1 / math.cos(center[0] * math.pi / 180.0)

  span = [lat_span, lng_span]

  return (center, span)

_COLORS = ['green', 'blue', 'red', 'yellow', 'purple']

def plot_points_on_map(points, tmp_image_dir='/tmp/mapimages', filename=None):
  """Given a set of points, plot them on a map using matplotlib.

  Args:
    points: A list of lists of points.  Each group of points will be plotted in a different
            color.  Each individual point is a 2-tuple of latitude and longitude.
    colors: A list of colors, one for each 
    image_dir: A directory in which to store images.  The temporary image downloaded from
               the Google Static Maps API will be stored here.
    filename: A file in which to store the plot itself.
  
  Returns:
    The Basemap object.
  """
  lats = [x[0] for group in points for x in group]
  lngs = [x[1] for group in points for x in group]
  print 'Plotting %d points' % len(lats)

  (center, span) = fit_center_span(lats, lngs)
  (north, south, east, west) = bounds(center, span)
  print '  center: %f,%f | span: %f,%f' % (center[0], center[1], span[0], span[1])

  plt.clf()
  m = Basemap(llcrnrlon=west, llcrnrlat=south, urcrnrlon=east, urcrnrlat=north,
              resolution='h', projection='merc', lat_ts=0,
              lon_0=center[1], lat_0=center[0])

  img_file = get_map_file(center, span, tmp_image_dir)
  img = plt.imread(img_file)
  im = m.imshow(img, interpolation='lanczos', origin='upper')

  for ii, group in enumerate(points):
    group_lats = [x[0] for x in group]
    group_lngs = [x[1] for x in group]
    color = _COLORS[ii % len(_COLORS)]

    x, y = m(group_lngs, group_lats)
    m.scatter(x, y, marker='.', c=color, lw=0.2)

  if filename:
    plt.savefig(filename, bbox_inches='tight')

  return m


################################################################
## Contour plotting
################################################################

class Gaussian:
  """Represents a single Gaussian shape and its components."""

  def __init__(self, mix, mu, covariance):
    """Initialize this Gaussian with the given mixing coefficient, mu, and covariance.

    Args:
      mix: The mixing coefficient.
      mu: A 2-tuple containing the x,y mean.
      covariance: A 3-tuple containing the elements of the covariance matrix, C11, C12 (=C21), C22.
    """
    self.mixing_coefficient = mix
    self.mu_x = mu[0]
    self.mu_y = mu[1]
    self.covariance_11 = covariance[0]
    self.covariance_12 = covariance[1]
    self.covariance_22 = covariance[2]

  def __str__(self):
    return ('\t\tmixing_coefficient: %f\n' % self.mixing_coefficient + \
            '\t\t              mu_x: %f\n' % self.mu_x + \
            '\t\t              mu_y: %f\n' % self.mu_y + \
            '\t\t     covariance_11: %f\n' % self.covariance_11 + \
            '\t\t     covariance_12: %f\n' % self.covariance_12 + \
            '\t\t     covariance_22: %f'   % self.covariance_22)

LOG_2PI = math.log(math.pi * 2.0)
RADIUS = 6367000.0  # Earth's radius in meters.

class VenueShape:
  def __init__(self, venueid, venuename, center, gaussians):
    """Create a venue shape from the given data

    Args:
      venueid: The venue's ID.
      venuename: The venue's name.
      center: A 2-tuple of the lat,lng for the center point of the venue.
      gaussians: A list of Gaussian objects making up the shape.
    """
    self.venueid = venueid
    self.components = gaussians
    self.center = center
    self.name = venuename

    # Stereographic projection compnents
    self.phi_1 = math.radians(self.center[0])
    self.lam_0 = math.radians(self.center[1])
    self.sin_phi_1 = math.sin(self.phi_1)
    self.cos_phi_1 = math.cos(self.phi_1)

    self.center_prob = self.log_density(self.center[0], self.center[1])

    self.vec_value = np.vectorize(self.value)

  def __str__(self):
    out = '%s (%s)\n\tCenter: (%f, %f)\n' % (self.name, self.venueid, self.center[0], self.center[1])
    return out + '\n'.join(['\tComponent %d:\n%s' % (ii, str(x)) for ii, x in enumerate(self.components)])

  def project(self, lat, lng):
    """Project the given lat,lng into stereographic space around the venue's center."""
    phi = math.radians(lat)
    lam = math.radians(lng)
    cos_phi = math.cos(phi)
    cos_lam_lam_0 = math.cos(lam - self.lam_0)
    sin_phi = math.sin(phi)
    k = (2.0 * RADIUS) / (1.0 + self.sin_phi_1 * sin_phi + self.cos_phi_1 * cos_phi * cos_lam_lam_0)

    return ((k * cos_phi * math.sin(lam - self.lam_0)),
            (k * (self.cos_phi_1 * sin_phi - self.sin_phi_1 * cos_phi * cos_lam_lam_0)))


  def log_density(self, lat, lng):
    """Return the value of this model at the given lat,lng."""
    (x, y) = self.project(lat, lng)

    def value_for_component(cmp):
      det = cmp.covariance_11 * cmp.covariance_22 - cmp.covariance_12 * cmp.covariance_12
      inv = (cmp.covariance_22 / det, -1.0 * cmp.covariance_12 / det, cmp.covariance_11 / det)
      if cmp.mixing_coefficient < 1e-10 or det < 1e-10:
        return -50.0
      else:
        xd = x - cmp.mu_x
        yd = y - cmp.mu_y
        quad = xd * xd * inv[0] + xd * yd * inv[1] + yd * yd * inv[2]
        return -0.5 * quad - 0.5 * math.log(det) - LOG_2PI + math.log(cmp.mixing_coefficient)

    cluster_values = [value_for_component(c) for c in self.components]
    m = np.max(cluster_values)
    def exp(v):
      if v - m < -50.0: return 0.0
      else: return math.exp(v - m)

    log_prob = m + math.log(np.sum([exp(v) for v in cluster_values]))
    return log_prob

  def value(self, lng, lat):
    return self.log_density(lat, lng)
  

def make_mesh(m):
  """Make a meshgrid appropriate for plotting on the given Basemap instance."""
  # TODO: derive this automatically based on size of plot.
  resolution = .0001
  minx, maxx = (m.llcrnrlon, m.urcrnrlon)
  miny, maxy = (m.llcrnrlat, m.urcrnrlat)
  x = np.arange(minx, maxx, resolution)
  y = np.arange(miny, maxy, resolution)
  return np.meshgrid(x, y)
  

def plot_one_shape(m, shape):
  """Plot a single venue shape on the given map.

  Args:
    m: A Basemap instance.
    shape: A VenueShape object to plot.
  """
  X, Y = make_mesh(m)
  Z = shape.vec_value(X, Y)
  contour = m.contour(X, Y, Z, latlon=True, linewidth=2.0,
                      levels=[-10, -5, -3, -1, 0],
                      colors=['blue', 'cyan', 'green', 'orange', 'red'])
  plt.clabel(contour, inline=1, fontsize=10)

def plot_two_shapes(m, shape1, shape2):
  """Plot two venue shapes on the given map, highlighting the boundary between the two.

  Args:
    m: A Basemap instance.
    shape1, shape2: VenueShape objects to plot.
  """
  X, Y = make_mesh(m)
  Z = shape1.vec_value(X, Y) - shape2.vec_value(X, Y)

  contour = m.contour(X, Y, Z, latlon=True, linewidth=2.0,
                      levels=[-10, -5, -3, -1, 0, 1, 3, 5, 10],
                      colors=['blue', 'cyan', 'green', 'orange', 'red', 'orange', 'green', 'cyan', 'blue'])
  plt.clabel(contour, inline=1, fontsize=10)
  m.contourf(X, Y, Z, latlon=True, alpha=0.2,
             levels=[-10, -5, -3, -1, 0, 1, 3, 5, 10],
             colors=['white', 'white', 'white', 'red', 'red', 'white', 'white', 'white'])
