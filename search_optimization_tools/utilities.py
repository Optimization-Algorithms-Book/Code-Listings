import math
'''
paths: List of Leaflet AntPaths

Output: The bounds [[South,West],[North,East]] of a map view that would show all the paths
'''
def get_paths_bounds(paths):
    lats = [p.locations[0][0] for p in paths]
    lngs = [p.locations[0][1] for p in paths]
    return [[min(lats),min(lngs)],[max(lats),max(lngs)]]

def straight_line(lon1, lat1, lon2, lat2):
    return math.sqrt((lon2 - lon1)**2 + (lat2-lat1)**2)

def haversine_distance(lon1, lat1, lon2, lat2):
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 

    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r