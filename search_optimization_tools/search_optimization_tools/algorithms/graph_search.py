import heapq
import math
from collections import deque
from ..structures import Solution
from time import process_time
from sys import getsizeof
from copy import deepcopy
import networkx as nx

"""
Requirements:
The node class must have the following public object methods:
- path(): generates a path from origin to this node by tracing node parents
- expand(): generates a list of all the children (reachable nodes) from this node
"""


def BFS(origin, destination):

    time_start = process_time()  # Time tracking
    max_frontier = 0  # Space tracking
    route = []
    frontier = deque([origin])
    explored = set()  # Explored tracking
    found = False

    while frontier and not found:
        node = frontier.popleft()
        explored.add(node)
        for child in node.expand():
            if child not in explored and child not in frontier:
                if child == destination:
                    route = child.path()
                    found = True
                frontier.append(child)
                if getsizeof(frontier) > max_frontier:
                    max_frontier = getsizeof(frontier)

    time_end = process_time()  # Time tracking

    return Solution(route, time_end - time_start, max_frontier, len(explored))


"""
Requirements:
The node class must have the following public object methods:
- path(): generates a path from origin to this node by tracing node parents
- expand(): generates a list of all the children (reachable nodes) from this node
"""


def DFS(origin, destination):
    time_start = process_time()  # Time tracking
    max_frontier = 0  # Space tracking

    route = []
    frontier = deque([origin])
    explored = set()  # Explored tracking
    found = False

    while frontier and not found:
        node = frontier.pop()
        explored.add(node)
        for child in node.expand():
            if child not in explored and child not in frontier:
                if child == destination:
                    route = child.path()
                    found = True
                frontier.append(child)
                if getsizeof(frontier) > max_frontier:
                    max_frontier = getsizeof(frontier)
    time_end = process_time()  # Time tracking
    return Solution(route, time_end - time_start, max_frontier, len(explored))


"""
Requirements:
The node class must have the following public object methods:
- path(): generates a path from origin to this node by tracing node parents
- expand(): generates a list of all the children (reachable nodes) from this node
- get_id(): returns a unique identifier for the node
- set_parent(node): sets the parent node of this node (this value will be used for the path method)
- get_distance(): returns the edge distance between this node and its parent
- set_distance(distance): sets the edge distance between this node and its parent


"""


def Dijkstra(origin, destination, unrelaxed_nodes):
    time_start = process_time()  # Time tracking
    space = getsizeof(deepcopy(unrelaxed_nodes))  # Space tracking

    # Using a set here avoids the problem with self loops
    seen = set()  # explored tracking
    shortest_dist = {node.get_id(): math.inf for node in unrelaxed_nodes}
    shortest_dist[origin.get_id()] = 0
    found = False
    route = None
    while len(unrelaxed_nodes) > 0 and not found:
        node = min(unrelaxed_nodes, key=lambda node: shortest_dist[node.get_id()])
        # relaxing the node, so this node's value in shortest_dist is the shortest distance between the origin and destination
        unrelaxed_nodes.remove(node)
        seen.add(node.get_id())
        # if the destination node has been relaxed then that is the route we want
        if node == destination:
            route = node.path()
            found = True
            continue
        # otherwise, let's relax edges of its neighbours
        for child in node.expand():
            # skip self-loops
            if child.get_id() in seen:
                continue
            child_obj = next(
                (node for node in unrelaxed_nodes if node.get_id() == child.get_id()),
                None,
            )
            child_obj.set_distance(child.get_distance())
            distance = shortest_dist[node.get_id()] + child.get_distance()
            if distance < shortest_dist[child_obj.get_id()]:
                shortest_dist[child_obj.get_id()] = distance
                child_obj.set_parent(node)
    time_end = process_time()  # Time tracking
    return Solution(route, time_end - time_start, space, len(seen))


# This implementation uses a heap with tuples (a,b),
# a is the cost of a node, and b is the node itself.
def UCS(origin, destination):
    time_start = process_time()  # Time tracking
    max_priority = 0  # Space tracking

    entry_count = 1
    priority_queue = [(0, 0, origin)]

    found = False
    visited = []  # Explored tracking
    while priority_queue and not found:
        node = heapq.heappop(priority_queue)
        node_cost = node[0]
        node = node[2]
        if node in visited:
            continue
        visited.append(node)
        # We found the destination
        if node == destination:
            route = node.path()
            found = True
            continue
        for child in node.expand():
            total_cost = child.get_distance() + node_cost
            matches = [item for item in priority_queue if item[2] == child]
            if matches:
                # Update the entry if the new priority is better
                if total_cost < matches[0][0]:
                    priority_queue[priority_queue.index(matches[0])] = (
                        total_cost,
                        entry_count,
                        child,
                    )
                    entry_count += 1
                    heapq.heapify(priority_queue)
            else:
                heapq.heappush(priority_queue, (total_cost, entry_count, child))
                if getsizeof(priority_queue) > max_priority:
                    max_priority = getsizeof(priority_queue)
                entry_count += 1
    time_end = process_time()  # Time tracking
    return Solution(route, time_end - time_start, max_priority, len(visited))


def Bidirectional_Dijkstra(origin, destination, unrelaxed_nodes, expand_kwargs = {}):
    time_start = process_time()  # Time tracking
    frontier = deepcopy(unrelaxed_nodes)

    space_required = getsizeof(frontier)

    explored_f = set()
    explored_b = set()

    shortest_dist_f = {node.get_id(): math.inf for node in frontier}
    shortest_dist_b = {node.get_id(): math.inf for node in frontier}

    shortest_dist_f[origin] = 0
    shortest_dist_b[destination] = 0

    found = False
    route = []

    altr_expand = True  # to alternate between front and back

    while frontier and not found:
        if altr_expand:  # Forward
            node = min(frontier, key=lambda node: shortest_dist_f[node.get_id()])
            # relaxing the node, so this node's value in shortest_dist is the shortest distance between the origin and destination
            frontier.remove(node)
            explored_f.add(node)
            # if the destination node has been relaxed then that is the route we want
            if node == destination:
                route = node.path()
                found = True
                continue

            # otherwise, let's relax edges of its neighbours
            for child in node.expand(**expand_kwargs):
                # skip self-loops
                if child.get_id() in explored_f:
                    continue

                # Check the child is collided
                if child in explored_b:
                    overlapped = next((node for node in explored_b if node == child))
                    # we don't take the overlapped node twice
                    route = child.path()[:-1] + overlapped.path()[::-1]
                    found = True
                    break

                child_obj = next(
                    (node for node in frontier if node.get_id() == child.get_id()), None
                )
                child_obj.set_distance(child.get_distance())
                distance = shortest_dist_f[node.get_id()] + child.get_distance()
                if distance < shortest_dist_f[child_obj.get_id()]:
                    shortest_dist_f[child_obj.get_id()] = distance
                    child_obj.set_parent(node)
            altr_expand = False
        if not altr_expand:  # Backward
            node = min(frontier, key=lambda node: shortest_dist_b[node.get_id()])
            # relaxing the node, so this node's value in shortest_dist is the shortest distance between the origin and destination
            frontier.remove(node)
            explored_b.add(node)
            # if the destination node has been relaxed then that is the route we want
            if node == origin:
                route = node.path()[::-1]
                found = True
                continue

            # otherwise, let's relax edges of its neighbours
            for child in node.expand(backwards=True, **expand_kwargs):
                # skip self-loops
                if child.get_id() in explored_b:
                    continue

                # Check the child is collided
                if child in explored_f:
                    overlapped = next(
                        (node for node in explored_f if node == child), None
                    )
                    route = overlapped.path()[:-1] + child.path()[::-1]
                    found = True
                    break

                child_obj = next(
                    (node for node in frontier if node.get_id() == child.get_id()), None
                )
                child_obj.set_distance(child.get_distance())
                distance = shortest_dist_b[node.get_id()] + child.get_distance()
                if distance < shortest_dist_b[child_obj.get_id()]:
                    shortest_dist_b[child_obj.get_id()] = distance
                    child_obj.set_parent(node)
            altr_expand = True
    time_end = process_time()  # Time tracking
    return Solution(
        route, time_end - time_start, space_required, len(explored_f) + len(explored_b)
    )


def Kruskal(G, attr="weight", sorted_edges=False, edges=None, graph_type=nx.Graph):
    if not sorted_edges:
        edges = sorted(G.edges(data=True), key=lambda t: t[2].get(attr, 1))
    mst = graph_type()
    mst.add_nodes_from(G)
    for e in edges:
        mst.add_edges_from([e])
        try:
            nx.find_cycle(mst)
            mst.remove_edge(e[0], e[1])
        except:
            try:
                if nx.is_connected(mst):
                    break
            except:
                if nx.is_strongly_connected(mst):
                    break
            continue
    return mst


"""
Requirements:
The node class must have the following public object methods:
- path(): generates a path from origin to this node by tracing node parents
- expand(): generates a list of all the children (reachable nodes) from this node
"""


def A_Star(
    G, origin, destination, heuristic_fn, heuristic_kwargs={}, expand_kwargs={}
):
    start_time = process_time()
    toDestination, toOrigin = heuristic_fn(G, origin, destination, **heuristic_kwargs)

    route = []
    frontier = list()
    frontier.append(origin)
    explored = set()
    found = False
    while frontier and not found:
        # choose a node based on its heuristic value
        node = min(frontier, key=lambda node: toOrigin[node.get_id()] + toDestination[node.get_id()])
        frontier.remove(node)
        explored.add(node)
        # expand its children
        for child in node.expand(**expand_kwargs):
            if child not in explored and child not in frontier:
                if child == destination:
                    route = child.path()
                    found = True
                    continue
                frontier.append(child)
    space = getsizeof(explored)
    end_time = process_time()
    return Solution(route, end_time-start_time, space, len(explored))
