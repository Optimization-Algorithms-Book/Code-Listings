import osmnx as ox
import folium
import folium.plugins
from .utilities import get_paths_bounds, straight_line
from .structures import Node
import copy
import math
import random
from collections import deque
import networkx as nx

'''
G: networkx.Graph, containing edges that have the 'length' attribute (most commonly osmnx graphs)
route: List of ints representing Node osmids from G that form a route. 

Output: The sum of the lengths of the edges in the route, rounded to 4 decimal places.
'''
def cost(G, route, attr_name='length'):
    weight = 0
    for u, v in zip(route, route[1:]):
        try:
            weight += G[u][v][0][attr_name]
        except:
            weight += G[u][v][attr_name]
    return round(weight,4)


'''
G: osmnx Graph
route: List of ints representing Node osmids from G that form a route. 
Output: Folium Map with Origin/Destination Markers and a Polyline or Antpath for the route
'''
def draw_route(G, route):    
    G_gdfs = ox.graph_to_gdfs(G)
    nodes_frame = G_gdfs[0]
    ways_frame = G_gdfs[1]
    start_node = nodes_frame.loc[route[0]]
    end_node = nodes_frame.loc[route[len(route)-1]]
    start_xy = (start_node['y'], start_node['x'])
    end_xy = (end_node['y'], end_node['x'])
    
    m = folium.Map(location = start_xy, zoom_start=15)
    folium.Marker(location = start_xy, draggable = False).add_to(m)
    folium.Marker(location = end_xy, draggable = False).add_to(m)

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
        folium.plugins.AntPath(
            locations = [*points], 
            dash_array=[1, 10],
            delay=1000,
            color='red',
            pulse_color='black'
        ).add_to(m)
    #     pathGroup.append(ant_path)
    #     m.add_layer(ant_path)
    # m.fit_bounds(get_paths_bounds(pathGroup))
    return m

def shortest_path_with_failed_nodes(G, route ,source, target, failed : list):
    origin = Node(graph = G, osmid = source)
    destination = Node(graph = G, osmid = target)

    ## you can't introduce failure in the source and target
    # node, because your problem will lose its meaning
    if source in failed: failed.remove(source)
    if target in failed: failed.remove(target)
    
    # if after removing source/target node from failed
    # list - just return math.inf which is equivalent to failure in search
    if len(failed) == 0: return math.inf

    # we need to flag every node whether it is failed or not
    failure_nodes = {node: False for node in G.nodes()}
    failure_nodes.update({node: True for node in failed})

    # we need to make sure that while expansion we don't expand
    # any node from the original graph to avoid loops in our route
    tabu_list = route[:route.index(source)] \
                + \
                route[route.index(target) + 1:] 

    # the normal implementation of dijkstra
    shortest_dist = {node: math.inf for node in G.nodes()}
    unrelaxed_nodes = [Node(graph = G, osmid = node) for node in G.nodes()]
    seen = set()

    shortest_dist[source] = 0

    while len(unrelaxed_nodes) > 0:
        node = min(unrelaxed_nodes, key = lambda node : shortest_dist[node])

        # if we have relaxed articulation nodes in our graph
        # halt the process -- we have more than one component
        # in our graph which makes the question of shortest path
        # invalid

        if shortest_dist[node.osmid] == math.inf: return math.inf

        if node == destination:
            return node.path()

        unrelaxed_nodes.remove(node); seen.add(node.osmid) # relaxing the node

        for child in node.expand():
            # if it is failed node, skip it
            if failure_nodes[child.osmid] or\
                child.osmid in seen or\
                child.osmid in tabu_list:
                continue

            child_obj = next((node for node in unrelaxed_nodes if node.osmid == child.osmid), None)
            child_obj.distance = child.distance

            distance = shortest_dist[node.osmid] + child.distance
            if distance < shortest_dist[child_obj.osmid]:
                shortest_dist[child_obj.osmid] = distance
                child_obj.parent = node

    # in case the node can't be reached from the origin
    # this return happens when the node is not on the graph
    # at all, if it was on a different component the second
    # return will be executed -- this is the third return
    
    return math.inf

def get_child(G, route):
    for i in range(1, len(route) - 1):
        for j in range(i, len(route) -1):
            # we can't work on the route list directly
            # because lists are passed by reference
            stitched = copy.deepcopy(route)
            failing_nodes = copy.deepcopy(route[i:j+1])
            to_be_stitched = shortest_path_with_failed_nodes(G, stitched, stitched[i-1], stitched[j+1], failing_nodes)
            
            # this would happen because one of the failing
            # nodes are articulation node and caused the graph
            # to be disconnected
            if to_be_stitched == math.inf: continue

            stitched[i:j+1] = to_be_stitched[1:-1]      # we need to skip the first and starting nodes of this route
                                                        # because these nodes already exit
            yield stitched

def randomized_search(G, source, destination):
    origin = Node(graph = G, osmid = source)
    destination = Node(graph = G, osmid = destination)
    
    route = [] # the route to be yielded
    frontier = deque([origin])
    explored = set()
    while frontier:
        node = random.choice(frontier)   # here is the randomization part
        frontier.remove(node)
        explored.add(node.osmid)

        for child in node.expand():
            if child not in explored and child not in frontier:
                if child == destination:
                    route = child.path()
                    return route
                frontier.append(child)

    raise Exception("destination and source are not on same component")


def astar_heuristic(G, origin, destination, measuring_dist = straight_line):
    distanceGoal = dict()
    distanceOrigin = dict()

    originX = G.nodes[origin.get_id()]['x']
    originY = G.nodes[origin.get_id()]['y']

    destX = G.nodes[destination.get_id()]['x']
    destY = G.nodes[destination.get_id()]['y']

    for node in G:
        pointX = G.nodes[node]['x']
        pointY = G.nodes[node]['y']

        originDist = measuring_dist(originX, originY, pointX, pointY)
        destDist = measuring_dist(pointX, pointY, destX, destY)

        distanceGoal[node] = originDist
        distanceOrigin[node] = destDist

    return distanceGoal, distanceOrigin


def generate_dijkstra(G, source, hierarchical_order, direction = 'up', weight='length'):

    # initializing 
    SP = dict()
    parent = dict()
    unrelaxed = list()
    for node in G.nodes():
        SP[node] = math.inf
        parent[node] = None
        unrelaxed.append(node)
    SP[source] = 0

    # dijkstra
    while unrelaxed:
        node = min(unrelaxed, key = lambda node : SP[node])
        print(node)
        unrelaxed.remove(node)
        if SP[node] == math.inf: break
        
        for child in G[node]:
            # skip unqualified edges
            if direction == 'up':
                if hierarchical_order[child] < hierarchical_order[node]: continue
            if direction == 'down':
                if hierarchical_order[child] > hierarchical_order[node]: continue

            # If we're building a down graph, we need to use reverse weights
            if direction == 'down':
                if node not in G[child]: continue
                distance = SP[node] + G[child][node][weight]
            else:
                distance = SP[node] + G[node][child][weight]
            if distance < SP[child]:
                SP[child] = distance
                parent[child] = node
    return parent, SP

def build_route(G,origin, destination, parent):
    if destination not in G[origin]:
        # We need the parent of the destination instead
        return build_route(G,origin,parent[destination], parent) + [destination]
    
    edge = G[origin][destination]
    if 'midpoint' in edge and 'osmid' not in edge: # This is a contracted edge
        before = build_route(G,origin,edge['midpoint'], parent)
        after = build_route(G,edge['midpoint'], destination, parent)
        return before[:-1] + after
    return [origin,destination]

def edge_differences(G,sp):
    edge_diffs = dict()
    seen_list = []

    for node in G.nodes:
        edges_incident = len(G[node])
        if edges_incident == 1: # Handle terminating points
            edge_diffs[node] = -1
            continue
        new_graph = G.copy()
        new_graph.remove_node(node)
        shortcuts = 0
        for neighbour in G[node]:
            for other_neighbour in G[node]:

                if neighbour == other_neighbour: continue
                if [neighbour,other_neighbour] in seen_list: continue
                seen_list.append([neighbour,other_neighbour])
                old_sp = sp[neighbour][other_neighbour]
                old_sp_rev = sp[other_neighbour][neighbour]
                try:
                    new_sp = nx.shortest_path_length(new_graph,neighbour,other_neighbour, weight='length')
                except:
                    new_sp = math.inf
                try:
                    new_sp_rev = nx.shortest_path_length(new_graph,other_neighbour,neighbour, weight='length')
                except:
                    new_sp_rev = math.inf
                need_new = old_sp != new_sp
                need_new_rev = old_sp_rev != new_sp_rev
                if need_new: shortcuts +=1
                if need_new_rev: shortcuts+=1
        ED = shortcuts - edges_incident
        edge_diffs[node] = ED
    return sorted(edge_diffs, key=lambda x:edge_diffs[x])

from tqdm.notebook import tqdm
def contract_graph(G: nx.DiGraph, edge_difference, sp):
    # to keep track of the edges added after the algorithm finishes

    seen_list = []

    for node in tqdm(edge_difference):
        edges_incident = len(G[node])
        if edges_incident == 1: continue # Terminating points

        new_graph = G.copy()
        new_graph.remove_node(node)

        for neighbour in G[node]:
            for other_neighbour in G[node]:

                if neighbour == other_neighbour: continue
                if [neighbour,other_neighbour] in seen_list: continue
                seen_list.append([neighbour,other_neighbour])
                old_sp = sp[neighbour][other_neighbour]
                old_sp_rev = sp[other_neighbour][neighbour]
                try:
                    new_sp = nx.shortest_path_length(new_graph,neighbour,other_neighbour, weight='length')
                except:
                    new_sp = math.inf
                try:
                    new_sp_rev = nx.shortest_path_length(new_graph,other_neighbour,neighbour, weight='length')
                except:
                    new_sp_rev = math.inf
                need_new = old_sp != new_sp
                need_new_rev = old_sp_rev != new_sp_rev

                if need_new:
                    G.add_edge(neighbour,other_neighbour,length=old_sp,midpoint=node)
                if need_new_rev:
                    G.add_edge(other_neighbour,neighbour,length=old_sp_rev,midpoint=node)