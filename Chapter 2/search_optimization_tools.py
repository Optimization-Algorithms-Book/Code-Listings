import ipyleaflet as lf
import folium as fl
import osmnx as ox
import networkx as nx

class Node:
    # using __slots__ for optimization
    __slots__ = ['node', 'distance', 'parent', 'osmid', 'G']
    # constructor for each node
    def __init__(self ,graph , osmid, distance = 0, parent = None):
        # the dictionary of each node as in networkx graph --- still needed for internal usage
        self.node = graph[osmid]
        
        # the distance from the parent node --- edge length
        self.distance = distance
        
        # the parent node
        self.parent = parent
        
        # unique identifier for each node so we don't use the dictionary returned from osmnx
        self.osmid = osmid
        
        # the graph
        self.G = graph
    
    # returning all the nodes adjacent to the node
    def expand(self):
        children = [Node(graph = self.G, osmid = child, distance = self.node[child][0]['length'], parent = self) \
                        for child in self.node]
        return children
    
    # returns the path from that node to the origin as a list and the length of that path
    def path(self):
        node = self
        path = []
        while node:
            path.append(node.osmid)
            node = node.parent
        return path[::-1]
    
    # the following two methods are for dictating how comparison works

    def __eq__(self, other):
        try:
            return self.osmid == other.osmid
        except:
            return self.osmid == other
            
    
    def __hash__(self):
        return hash(self.osmid)

def cost(G, route):
    weight = 0
    for u, v in zip(route, route[1:]):
        weight += G[u][v][0]['length']   
    return round(weight,4)


def get_center(G):
    undir = G.to_undirected()
    length_func = nx.single_source_dijkstra_path_length
    sp = {source: dict(length_func(undir, source, weight="length")) for source in G.nodes}
    eccentricity = nx.eccentricity(undir,sp=sp)
    center_osmid = nx.center(undir,e=eccentricity)[0]
    return center_osmid

def get_paths_bounds(paths):
    lats = [p.locations[0][0] for p in paths]
    lngs = [p.locations[0][1] for p in paths]
    return [[min(lats),min(lngs)],[max(lats),max(lngs)]]


def draw_route(G, route, zoom = 15, force_leaflet=False):
    
    center_osmid = get_center(G)
    G_gdfs = ox.graph_to_gdfs(G)
    nodes_frame = G_gdfs[0]
    ways_frame = G_gdfs[1]
    center_node = nodes_frame.loc[center_osmid]
    location = (center_node['y'], center_node['x'])
    

    start_node = nodes_frame.loc[route[0]]
    end_node = nodes_frame.loc[route[len(route)-1]]

    start_xy = (start_node['y'], start_node['x'])
    end_xy = (end_node['y'], end_node['x'])
    

    if len(route) >= 500 and not force_leaflet:
        print(f"The route has {len(G)} elements, using folium to improve performance.")
        m = ox.plot_route_folium(G = G, route = route, zoom= zoom, color='red')
        fl.Marker(location=start_xy).add_to(m)
        fl.Marker(location=end_xy).add_to(m)
        return m

    m = lf.Map(center = location, zoom = zoom)
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