'''
paths: List of Leaflet AntPaths

Output: The bounds [[South,West],[North,East]] of a map view that would show all the paths
'''
def get_paths_bounds(paths):
    lats = [p.locations[0][0] for p in paths]
    lngs = [p.locations[0][1] for p in paths]
    return [[min(lats),min(lngs)],[max(lats),max(lngs)]]