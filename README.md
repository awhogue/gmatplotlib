gmatplotlib
===========

Libraries for working with <a href="http://matplotlib.org">matplotlib</a> and the <a href="https://developers.google.com/maps/documentation/staticmaps/">Google Static Maps API</a>.

Example usage:

    import gmatplotlib as gm
    # Points should look like [[(40.7, -72.0), (41.3, -72.2), ...],
                               [(37.78, -122.40), (37.8, -122.39)]]
    points = ...
    m = gm.plot_points_on_map(points, output_filename='myfig.png')
    shape = gm.VenueShape('1234', 'Foursquare HQ', (23, 45), (12, 34, 56))
    gm.plot_one_shape(m, shape1)
