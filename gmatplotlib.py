#!/usr/bin/python

import random, sys, datetime, time, os, math
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np

"""
Libraries for working with matplotlib and Google Maps.
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

_COLORS = ['green', 'blue', 'red', 'purple']

def plot_points_on_map(points, tmp_image_dir='/tmp/mapimages', filename=None):
  """Given a set of points, plot them on a map using matplotlib.

  Args:
    points: A list of lists of points.  Each group of points will be plotted in a different
            color.  Each individual point is a 2-tuple of latitude and longitude.
    colors: A list of colors, one for each 
    image_dir: A directory in which to store images.  The temporary image downloaded from
               the Google Static Maps API will be stored here.
    filename: A file in which to store the plot itself.
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
    plt.savefig(filename)
