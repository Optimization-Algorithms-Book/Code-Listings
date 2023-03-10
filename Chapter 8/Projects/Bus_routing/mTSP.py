def runTSP(acs, edge):
    acs.reset_sub_edges(edge)
    acs.start()
    return acs.global_best_distance, acs.actual_global_best_tour


def run(pool, ACS_list, cluster):
    results = [pool.apply_async(runTSP,args=(ACS_list[i], cluster[i])) for i in range(len(cluster))]
    routes = []
    total_dist = 0
    for r in results:
        total_dist += r.get()[0]
        routes.append(r.get()[1])
    return total_dist, routes
