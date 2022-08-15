import ipyleaflet as lf
import osmnx as ox
from .utilities import get_paths_bounds

'''
G: networkx.Graph, containing edges that have the 'length' attribute (most commonly osmnx graphs)
route: List of ints representing Node osmids from G that form a route. 

Output: The sum of the lengths of the edges in the route, rounded to 4 decimal places.
'''
def cost(G, route, attr_name='length', multigraph=True):
    weight = 0
    for u, v in zip(route, route[1:]):
        weight += G[u][v][0][attr_name] if multigraph else  G[u][v][attr_name]
    return round(weight,4)


'''
G: osmnx Graph
route: List of ints representing Node osmids from G that form a route. 
force_leaflet: Override flag to force using leaflet even for large graphs (affects performance)

Output: Leaflet or Folium Map with Origin/Destination Markers and a Polyline or Antpath for the route
'''
def draw_route(G, route):
    
    G_gdfs = ox.graph_to_gdfs(G)
    nodes_frame = G_gdfs[0]
    ways_frame = G_gdfs[1]
    start_node = nodes_frame.loc[route[0]]
    end_node = nodes_frame.loc[route[len(route)-1]]
    start_xy = (start_node['y'], start_node['x'])
    end_xy = (end_node['y'], end_node['x'])
    
    m = lf.Map(center = start_xy)
    marker = lf.Marker(location = start_xy, draggable = False)
    m.add_layer(marker)
    marker = lf.Marker(location = end_xy, draggable = False)
    m.add_layer(marker)
    pathGroup = []
    for u, v in zip(route[0:], route[1:]):
        try:
            geo = (ways_frame.query(f'u == {u} and v == {v}').to_dict('list')['geometry'])
            m_geo = min(geo,key=lambda x:x.length)
        except:
            geo = (ways_frame.query(f'u == {v} and v == {u}').to_dict('list')['geometry'])
            m_geo = min(geo,key=lambda x:x.length)
        x, y = m_geo.coords.xy
        points = map(list, [*zip([*y],[*x])])
        ant_path = lf.AntPath(
            locations = [*points], 
            dash_array=[1, 10],
            delay=1000,
            color='red',
            pulse_color='black'
        )
        pathGroup.append(ant_path)
        m.add_layer(ant_path)
    m.fit_bounds(get_paths_bounds(pathGroup))
    return m