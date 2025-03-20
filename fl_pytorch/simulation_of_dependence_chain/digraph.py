#!/usr/bin/env python

import matplotlib.pyplot as plt
from pygments.lexer import default


class DiEdge:
    def __init__(self):
        self.source = 0
        self.destination = 0
        self.cost = 0

class DiGraph:
    def __init__(self):
        self.V = set()
        self.E = set()
        self.EdgesPerVertex = dict()

    def clean(self):
        self.V = set()
        self.E = set()
        self.EdgesPerVertex = dict()

    def addVertex(self, v):
        self.V.add(v)

    def removeVertex(self, v):

        for s in self.V:
            i = len(self.EdgesPerVertex[s]) - 1
            while i >= 0:
                e = self.EdgesPerVertex[s][i]
                if e.destination == v or e.source == v:
                    self.E.remove(e)
                    del self.EdgesPerVertex[s][i]
                i -= 1

        self.V.remove(v)
        del self.EdgesPerVertex[v]

    def hasVertex(self, v):
        return v in self.V

    def addEdge(self, source, destination, cost = None):
        #===============================================================================================================
        if not self.hasVertex(source):
            self.addVertex(source)
        if not self.hasVertex(destination):
            self.addVertex(destination)
        #===============================================================================================================
        if source not in self.EdgesPerVertex:
            self.EdgesPerVertex[source] = list()
        if destination not in self.EdgesPerVertex:
            self.EdgesPerVertex[destination] = list()
        #===============================================================================================================
        diedge = DiEdge()
        diedge.source      = source
        diedge.destination = destination
        diedge.cost        = cost
        self.EdgesPerVertex[source].append(diedge)
        self.E.add(diedge)

    def topologicalSort(self):
        marked = {}
        for v in self.V:
            marked[v] = False

        processedList = []

        for v in self.V:
            if not marked[v]:
                self.dfsForTopSort(v, marked, processedList)

        processedList.reverse()
        return processedList

    def dfsForTopSort(self, v, marked, processedList):
        marked[v] = True

        for e in self.EdgesPerVertex[v]:
            if not marked[e.destination]:
                self.dfsForTopSort(e.destination, marked, processedList)

        processedList.append(v)
        return

    def isDag(self):
        marked = {}
        in_stack = {}

        for v in self.V:
            marked[v] = False
            in_stack[v] = False

        for v in self.V:
            if not marked[v]:
                if not self.dfsForDag(v, marked, in_stack):
                    # print("CYCLE:", [k for k, v in in_stack.items() if v == True])
                    return False

        return True

    def dfsForDag(self, v, marked, in_stack):
        marked[v] = True
        in_stack[v] = True

        for e in self.EdgesPerVertex[v]:
            if not marked[e.destination]:
                if not self.dfsForDag(e.destination, marked, in_stack):
                    # Cycle is detected
                    return False
            else:
                if in_stack[e.destination]:
                    # Cycle is detected
                    return False

        in_stack[v] = False
        return True

    def acyclicShortestPaths(self, sourceVertex):
        distTo = {}
        edgeTo = {}

        if not self.hasVertex(sourceVertex):
            return False, distTo, edgeTo

        if not self.isDag():
            return False, distTo, edgeTo

        for v in self.V:
            distTo[v] = float('inf')
        distTo[sourceVertex] = 0.0


        topOrder = self.topologicalSort()

        for v in topOrder:
            for e in self.EdgesPerVertex[v]:
                w = e.destination
                if distTo[w] > distTo[v] + e.cost:
                    distTo[w] = distTo[v] + e.cost
                    edgeTo[w] = e

        return True, distTo, edgeTo

    def dumpToDot(self, fname):
        fs = open(fname, "w")
        fs.write('digraph G {\n')

        for v in self.V:
            fs.write(f'\"{v}\"[shape = box];\n')

        for v in self.V:
            for e in self.EdgesPerVertex[v]:
                if abs(e.cost) <= 0.0:
                    fs.write(f'\"{e.source}\" -> \"{e.destination}\" [style=dotted, color=blue, label=\"\"];\n')
                else:
                    fs.write(f'\"{e.source}\" -> \"{e.destination}\" [color=blue, label=\" {e.cost:.3g} \"];\n')
        fs.write('}\n')
        fs.close()

    def dumpToDotWithCriticalPath(self, fname, timeTo, dependecies):
        fs = open(fname, "w")
        fs.write('digraph G {\n')

        for v in self.V:
            style = ""

            if "at master" in v.lower():
                # Fill master computation with orange
                style = "style=filled, fillcolor=orange,"
            elif "=>" in v:
                # Fill master computation with orange
                if "=> master" in v.lower():
                    style = "style=filled, fillcolor=yellow,"
                else:
                    style = f"style=filled, fillcolor=green,"

            if "compute g(xi)" in v.lower():
                style = "style=filled, fillcolor=grey,"

            fs.write(f'\"{v}\"[{style}shape = box, label=\"{v} || start:{abs(timeTo[v]):.3f}\"];\n')

        for v in self.V:
            used_color = "blue"
            for e in self.EdgesPerVertex[v]:

                if e.destination in dependecies:
                    critical_edge = dependecies[e.destination]
                    if e == critical_edge:
                        used_color = "red"

                if abs(e.cost) <= 0.0:
                    fs.write(f'\"{e.source}\" -> \"{e.destination}\" [style=dotted, color={used_color}, label=\"\"];\n')
                else:
                    fs.write(f'\"{e.source}\" -> \"{e.destination}\" [color={used_color}, label=\" {e.cost:.3g} \"];\n')
        fs.write('}\n')
        fs.close()

def test_digraph():
    g = DiGraph()
    g.addVertex("1 1")
    g.addVertex("2")
    g.addVertex("2")
    g.addEdge("1 1", "2", 12.4)

def test_digraph_dumping():
    g = DiGraph()
    g.addEdge("2 2", "3", 12.4)
    g.addEdge("1", "2 2", 12.4)
    g.dumpToDot("my.txt")

def test_topological_sort_and_ac_sh_paths():

    g = DiGraph()
    g.addEdge("2", "3", 12.4)
    g.addEdge("1", "2", 12.4)
    assert g.isDag()

    z = g.topologicalSort()
    assert z == ['1', '2', '3']


    g.clean()
    g.addEdge("1", "2", 12.4)
    g.addEdge("2", "3", 12.4)

    assert g.isDag()
    z = g.topologicalSort()
    assert z == ['1', '2', '3']

    g.clean()
    g.addEdge("2", "3", 10)
    g.addEdge("1", "2", 1)
    g.addEdge("3", "6", 100)

    g.addEdge("1", "4", 2)
    g.addEdge("4", "5", 3)
    g.addEdge("5", "6", 8)

    g.addEdge("6", "7", -2)

    assert g.isDag()
    z = g.topologicalSort()

    assert z[+0] == '1'
    assert z[-1] == '7'
    assert z[-2] == '6'

    resFinAcylc, distTo, edgeTo = g.acyclicShortestPaths("1")

    assert resFinAcylc == True
    assert distTo['2'] == 1.0
    assert distTo['4'] == 2.0
    assert distTo['3'] == 11.0
    assert distTo['7'] == 11.0

    g.clean()
    g.addEdge("1", "2")
    g.addEdge("2", "3")
    g.addEdge("2", "4")
    g.addEdge("3", "1")
    assert g.isDag() == False

def test_create_and_remove():
   g = DiGraph()
   g.addEdge("3", "1", 12.4)
   g.addEdge("2", "3", 12.4)
   g.addEdge("1", "2", 12.4)
   g.addEdge("3", "3", 12.4)

   g.removeVertex("3")
   assert len(g.V) == 2 and len(g.E) == 1 and len(g.EdgesPerVertex) == 2

   g.removeVertex("2")
   assert len(g.V) == 1 and len(g.E) == 0 and len(g.EdgesPerVertex) == 1

   g.removeVertex("1")
   assert len(g.V) == 0 and len(g.E) == 0 and len(g.EdgesPerVertex) == 0



def parallelJobScheduling(g):
    # g contains precedence constrains and cost is duration

    # Step-1 - change sign for all costs
    for e in g.E:
        e.cost = -e.cost

    # Step-2 - add fake source and sink
    srcName = "fake_source"
    sinkName = "fake_sink"

    g.addVertex(srcName)
    g.addVertex(sinkName)

    for v in g.V:
        if v == srcName or v == sinkName:
            continue

        g.addEdge(srcName, v, 0.0)
        g.addEdge(v, sinkName, 0.0)

    # Step-3 - find longest path from fake source
    resFinAcylc, distTo, edgeTo = g.acyclicShortestPaths(srcName)
    if not resFinAcylc:
        return False, distTo, edgeTo
    else:
        for v in g.V:
            if v == srcName or v == sinkName:
                continue
            else:
                distTo[v] = -distTo[v]

    # Step-4 - remove fake source and sink with all outgoing and incoming links
    g.removeVertex(srcName)
    g.removeVertex(sinkName)

    # Step-5 - change sign for all costs again
    for e in g.E:
        e.cost = -e.cost

    return resFinAcylc, distTo, edgeTo

def test_parallel_job_scheduling():
    g = DiGraph()
    g.addEdge("compute_g_in_client_1", "send_g_from_client_1", 10.0)
    g.addEdge("send_g_from_client_1", "accept_by_master", 12.0)

    g.addEdge("compute_g_in_client_2", "send_g_from_client_2", 12.0)
    g.addEdge("send_g_from_client_2", "accept_by_master", 2.0)

    g.addEdge("accept_by_master", "compute_avg", 11.0)
    g.addEdge("compute_avg", "broadcast", 1.0)

    res, times, dependecies = parallelJobScheduling(g)
    assert times['accept_by_master'] == 22.0
    assert times['compute_avg'] == 33.0
    assert times['broadcast'] == 34.0

    assert res == True

#===============================
# CONFIGURATION
mult_samples_num = 1000
#===============================
def simulationGD(d, n, reestimate = False, times_prev = None, dependecies_prev = None, g_prev = None):
    g = DiGraph()

    rounds = 4
    bits_per_component_value = 32
    bits_per_index_value     = 32

    include_local_memory_overhead = True

    # various costs per client
    for r in range(rounds):
        #===============================================================================================================
        # Intel-xeon-e5-2666-v3
        clock_rate = 3.2*(10**9)        # Assume we have CPU CLock 3.2 GHz
        number_of_cores = 10            # Number of cores
        hyperthreading  = 2             # Hyperthreading
        ports_with_fpu_units = 2        # Number of FPU units
        mad_is_possible      = True     # MAD is possible
        ports_with_load_store_units = 3 # Number of Load/Store Units per core

        peak_flops = clock_rate * number_of_cores * hyperthreading * ports_with_fpu_units
        if mad_is_possible:
            peak_flops = peak_flops * 2

        # Assumption: 1 closk per add or subtract, 1 clock per multiply, with throughput 1operation/clock.
        #   Approximately the true if look into characterstic of vectorized multiply add operations VFMADDPS in x86
        #   https://www.agner.org/optimize/instruction_tables.pdf , page 46

        addition_costs = [1.0 / peak_flops] * n   # Price in cloks converted to seconds
        mult_costs = [1.0 / peak_flops] * n       # Price in cloks converted to seconds

        # Number of samples
        samples_in_clients = [55 * mult_samples_num] + [11 * mult_samples_num] * (n - 1)

        # Substitute average across the globe: https://www.speedtest.net/global-index

        # bits/sec (MBps => bits/second)
        bandwith_c_to_m = 41.54 * (10**6)    # uplink (client=>master)
        bandwith_m_to_c = 41.54 * (10 ** 6)  # downlink  (master=>client)

        rtts = [28*(10**-3)] * n             # rtt time (milliseconds => seconds)

        # Turn of rtts
        #for i in range(len(rtts)): rtts[i] = 0.0

        # Make compute cheaper
        #for i in range(len(addition_costs)): addition_costs[i] /= 100000.0
        #for i in range(len(mult_costs)): mult_costs[i] /= 100000.0

        cache_line_size_in_bits = 64 * 8

        # Another memory access we currently ignore:
        #   L1 access 4 cycles
        l1_access = 4 / clock_rate / ports_with_load_store_units / number_of_cores / (cache_line_size_in_bits) * bits_per_component_value
        #   L2 access 10 cycles
        l2_access = 10 / clock_rate / ports_with_load_store_units / number_of_cores / (cache_line_size_in_bits) * bits_per_component_value
        #   L3 access 40 cycles
        l3_access = 40 / clock_rate / ports_with_load_store_units / number_of_cores / (cache_line_size_in_bits) * bits_per_component_value
        #   Main memory access depend on memory type installation, but for Intel Core series it's approximately:
        # https://www.intel.com/content/www/us/en/support/articles/000056722/processors/intel-core-processors.html
        ddr4_memory_access = 1.0 / (2933 * 2 * 8 * 4) / (cache_line_size_in_bits) * bits_per_component_value

        # ===============================================================================================================
        if include_local_memory_overhead:
            l1_access = 0.0
            l2_access = 0.0
            l3_access = 0.0
            ddr4_memory_access = 0.0
        # ===============================================================================================================
        mem_latency = l2_access * 2 # For two operands
        #===============================================================================================================

        for i in range(n):
            addition_cost = addition_costs[i]
            mult_cost = mult_costs[i]
            samples_in_client = samples_in_clients[i]

            rtt = rtts[i]

            c= f"c{i + 1},k={r}"

            g.addEdge(f"Compute g(xi) at {c} [start]", f"Eval Ai@x at {c} [start]", 0.0)
            g.addEdge(f"Eval Ai@x at {c} [start]", f"Eval Ai@x - b at {c} [start]", samples_in_client * (d-1) * addition_cost + samples_in_client * d * mult_cost +
                                                                                    samples_in_client * d * mem_latency +  1 * d * mem_latency)

            g.addEdge(f"Eval Ai@x - b at {c} [start]", f"Eval A^T(Ai@x - b) at {c} [start]", d * addition_cost + d * mem_latency)

            g.addEdge(f"Eval A^T(Ai@x - b) at {c} [start]",  f"Scale A^T(Ai@x - b) at {c} [start]", d * (samples_in_client-1) * addition_cost + d * samples_in_client * mult_cost +
                                                                                                    d * samples_in_client * mem_latency +  1 * samples_in_client * mem_latency)

            g.addEdge(f"Scale A^T(Ai@x - b) at {c} [start]", f"Send at {c} => master[start]", d * mult_cost + d * mem_latency)

            # Assume NIC read data via DMA directly from Main Memory
            if reestimate:
                start_prev = times_prev[f"Send at {c} => master[start]"]
                end_prev   = times_prev[f"Sync at Master in k={r} [start]"]
                assert end_prev >= start_prev

                all_sends_to_master = [e for e in g_prev.E if "=> master" in e.source and f"{c}" not in e.source]
                intersections = 0
                for e in all_sends_to_master:
                    s_old = times_prev[e.source]
                    e_old = times_prev[e.source] + e.cost

                    if s_old > end_prev or e_old < start_prev:
                        continue
                    else:
                        start_intersection = max(s_old, start_prev)
                        end_intersection   = min(e_old, end_prev)
                        intersections += (end_intersection - start_intersection) / (end_prev - start_prev)

                # Totally there are only intersections + 1 from "n" which utilize communication bus
                multiplier = n/(intersections + 1)
                g.addEdge(f"Send at {c} => master[start]", f"Sync at Master in k={r} [start]", rtt/2.0 + (d*bits_per_component_value)/(bandwith_c_to_m * multiplier) + d * l3_access)
            else:
                g.addEdge(f"Send at {c} => master[start]", f"Sync at Master in k={r} [start]", rtt/2.0 + (d*bits_per_component_value)/bandwith_c_to_m + d * l3_access)

        g.addEdge(f"Sync at Master in k={r} [start]", f"Compute Global Gradient at Master in k={r} [start]", 0)
        g.addEdge(f"Compute Global Gradient at Master in k={r} [start]", f"Ready to broadcast at Master in k={r} [start]", n * d * (addition_cost+mem_latency) + 1.0 * d * (mult_cost+mem_latency) ) # avergaing
        g.addEdge(f"Ready to broadcast at Master in k={r} [start]", f"Ready to broadcast at Master in k={r} [end]", 0)
        # Master just send to everybody step to apply

        for i in range(n):
            addition_cost = addition_costs[i]
            mult_cost = mult_costs[i]
            samples_in_client = samples_in_clients[i]
            rtt = rtts[i]

            c= f"c{i + 1},k={r}"
            cNext = f"c{i + 1},k={r+1}"

            g.addEdge(f"Ready to broadcast at Master in k={r} [end]", f"At {c} obtaining update, master=>{c} [start]", 0)

            if reestimate:
                start_prev = times_prev[f"At {c} obtaining update, master=>{c} [start]"]
                end_prev   = times_prev[f"At {c} make a step [start]"]
                assert end_prev >= start_prev

                all_sends_from_master = [e for e in g_prev.E if "master=>" in e.source and f"{c}" not in e.source]

                intersections = 0.0

                for e in all_sends_from_master:
                    s_old = times_prev[e.source]
                    e_old = times_prev[e.source] + e.cost

                    if s_old > end_prev or e_old < start_prev:
                        continue
                    else:
                        start_intersection = max(s_old, start_prev)
                        end_intersection   = min(e_old, end_prev)
                        intersections += (end_intersection - start_intersection) / (end_prev - start_prev)

                # Totally there are only intersections + 1 from "n" which utilize communication bus
                multiplier = n/(intersections + 1)
                g.addEdge(f"At {c} obtaining update, master=>{c} [start]", f"At {c} make a step [start]", rtt / 2.0 + (d * bits_per_component_value) / (bandwith_m_to_c * multiplier) + d * l3_access)
            else:
                g.addEdge(f"At {c} obtaining update, master=>{c} [start]", f"At {c} make a step [start]", rtt/2.0 + (d*bits_per_component_value)/bandwith_m_to_c + d * l3_access)

            g.addEdge(f"At {c} make a step [start]", f"At {c} make a step [end]", d * addition_cost + 1.0 * d * mult_cost + d * mem_latency)        # step and scaling

            if r + 1 < rounds:
                # Dependence for next iteration
                g.addEdge(f"At {c} make a step [end]", f"Compute g(xi) at {cNext} [start]", 0)

    # Make parallel Job Scheduling
    res, times, dependecies = parallelJobScheduling(g)
    assert res == True

    return res, times, dependecies, g

def timeLineForBlock(show_time_line, d, n, algname, times, deps, g, block, uplink, downlink, computation):
    communication = uplink or downlink

    # Declaring a bar in schedule
    master_segments = []
    master_segments_communication_from_master = []
    master_segments_communication_to_master   = []

    client_segments = {}
    client_segments_communication_from_master = {}
    client_segments_communication_to_master   = {}

    for i in range(n):
        client_segments[i] = []
        client_segments_communication_from_master[i] = []
        client_segments_communication_to_master[i] = []

    max_time = 0.0

    for e in g.E:
        src = e.source
        dst = e.destination
        #if abs(e.cost) <= 0.0:
        #    continue

        source_name = src.lower()

        # ==========================================================================================================
        ignore_src = False
        for ii in range(n):
            if f"b={ii + 1}" in source_name or f"b = {ii + 1}" in source_name or f"block={ii + 1}" in source_name or f"block {ii + 1}" in source_name:
                if ii + 1 != block:
                    ignore_src = True
                    break

        if ignore_src:
            continue
        # ==========================================================================================================

        found_client = False
        for i in range(n):
            if f"at c{i+1}" in source_name:
                if "=>" in source_name:
                    if "=> master" in source_name.lower():
                        if uplink:
                            client_segments_communication_to_master[i].append((times[src], e.cost))
                    else:
                        if downlink:
                            client_segments_communication_from_master[i].append((times[src], e.cost))
                else:
                    client_segments[i].append((times[src], e.cost))

                found_client = True
                break

        if not found_client:
            if "=>" in source_name:
                if "=> master" in source_name.lower():
                    if uplink:
                        master_segments_communication_to_master.append((times[src], e.cost))
                else:
                    if downlink:
                        master_segments_communication_from_master.append((times[src], e.cost))
            else:
                master_segments.append((times[src], e.cost))

        # Update max time
        max_time = max(max_time, times[src] + e.cost)

    print(f"Time for all process ({algname},d={d},n={n}): ", max_time)

    if show_time_line:
        fontSize = 37
        plt.rc('xtick', labelsize=fontSize)
        plt.rc('ytick', labelsize=fontSize)

        # Setting X-axis limits
        fig, gnt = plt.subplots()

        # Setting Y-axis limits
        gnt.set_ylim(0, 50)

        # Setting labels for x-axis and y-axis
        gnt.set_xlabel('Seconds', fontdict = {'fontsize' : fontSize})
        gnt.set_ylabel('')

        # Setting ticks on y-axis
        gnt.set_yticks([15 + 10*i for i in range(n + 1)])
        # Labelling tickes of y-axis
        gnt.set_yticklabels(['Master'] + ["c" + str(i + 1) for i in range(n)])

        # Setting graph attribute
        #gnt.grid(True)

        gnt.set_xlim(0, max_time)
        #plt.title(f'{algname}, n={n},d={d},block={block}', fontsize=fontSize)

        if master_segments and computation:
            gnt.broken_barh(master_segments, (10, 9), facecolors='tab:orange',linewidth=1,edgecolor='black',linestyle='dashed')
        if master_segments_communication_to_master and communication:
            gnt.broken_barh(master_segments_communication_to_master, (10, 9), facecolors='yellow',linewidth=1,edgecolor='black',linestyle='dashed')
        if master_segments_communication_from_master and communication:
            gnt.broken_barh(master_segments_communication_from_master, (10, 9), facecolors='green',linewidth=1,edgecolor='black',linestyle='dashed')

        for i in range(n):
            if client_segments[i] and computation:
                gnt.broken_barh(client_segments[i], (20 + 10*i, 9), linewidth=1,edgecolor='black',linestyle='dashed')
            if client_segments_communication_to_master[i] and communication:
                gnt.broken_barh(client_segments_communication_to_master[i], (20 + 10*i, 9), linewidth=1,edgecolor='black',facecolors='yellow',linestyle='dashed')
            if client_segments_communication_from_master[i] and communication:
                gnt.broken_barh(client_segments_communication_from_master[i], (20 + 10*i, 9), linewidth=1,edgecolor='black',facecolors='green',linestyle='dashed')

        plt.show()

def timeLine(show_time_line, d, n, algname, times, deps, g):
    # Declaring a bar in schedule
    master_segments = []
    master_segments_communication_from_master = []
    master_segments_communication_to_master   = []

    client_segments = {}
    client_segments_communication_from_master = {}
    client_segments_communication_to_master   = {}

    for i in range(n):
        client_segments[i] = []
        client_segments_communication_from_master[i] = []
        client_segments_communication_to_master[i] = []

    max_time = 0.0

    for e in g.E:
        src = e.source
        dst = e.destination
        #if abs(e.cost) <= 0.0:
        #    continue

        source_name = src.lower()

        found_client = False
        for i in range(n):
            if f"at c{i+1}" in source_name:
                if "=>" in source_name:
                    if "=> master" in source_name.lower():
                        client_segments_communication_to_master[i].append((times[src], e.cost))
                    else:
                        client_segments_communication_from_master[i].append((times[src], e.cost))
                else:
                    client_segments[i].append((times[src], e.cost))

                found_client = True
                break

        if not found_client:
            if "=>" in source_name:
                if "=> master" in source_name.lower():
                    master_segments_communication_to_master.append((times[src], e.cost))
                else:
                    master_segments_communication_from_master.append((times[src], e.cost))
            else:
                master_segments.append((times[src], e.cost))

        # Update max time
        max_time = max(max_time, times[src] + e.cost)

    print(f"Time for all process ({algname},d={d},n={n}): ", max_time)

    if show_time_line:
        fontSize = 37
        plt.rc('xtick', labelsize=fontSize)
        plt.rc('ytick', labelsize=fontSize)

        # Setting X-axis limits
        fig, gnt = plt.subplots()

        # Setting Y-axis limits
        gnt.set_ylim(0, 50)

        # Setting labels for x-axis and y-axis
        gnt.set_xlabel('Seconds', fontdict = {'fontsize' : fontSize})
        gnt.set_ylabel('')

        # Setting ticks on y-axis
        gnt.set_yticks([15 + 10*i for i in range(n + 1)])
        # Labelling tickes of y-axis
        gnt.set_yticklabels(['Master'] + ["C" + str(i + 1) for i in range(n)])

        # Setting graph attribute
        #gnt.grid(True)

        gnt.set_xlim(0, max_time)

        #plt.title(f'{algname}, n={n},d={d}', fontsize=fontSize)

        if master_segments:
            gnt.broken_barh(master_segments, (10, 9), facecolors='tab:orange',linewidth=1,edgecolor='black',linestyle='dashed',alpha=0.5)
        if master_segments_communication_to_master:
            gnt.broken_barh(master_segments_communication_to_master, (10, 9), facecolors='yellow',linewidth=1,edgecolor='black',linestyle='dashed',alpha=0.5)
        if master_segments_communication_from_master:
            gnt.broken_barh(master_segments_communication_from_master, (10, 9), facecolors='green',linewidth=1,edgecolor='black',linestyle='dashed',alpha=0.5)

        for i in range(n):
            if client_segments[i]:
                gnt.broken_barh(client_segments[i], (20 + 10*i, 9), linewidth=1,edgecolor='black',linestyle='dashed',alpha=0.5)
            if client_segments_communication_to_master[i]:
                gnt.broken_barh(client_segments_communication_to_master[i], (20 + 10*i, 9), linewidth=1,edgecolor='black',facecolors='yellow',linestyle='dashed',alpha=0.5)
            if client_segments_communication_from_master[i]:
                gnt.broken_barh(client_segments_communication_from_master[i], (20 + 10*i, 9), linewidth=1,edgecolor='black',facecolors='green',linestyle='dashed',alpha=0.5)

        plt.show()

def simulationPermKSingleMaster(d, n, bandwithDivider, reestimate = False, times_prev = None, dependecies_prev = None, g_prev = None):
    g = DiGraph()

    rounds = 4
    bits_per_component_value = 32
    bits_per_index_value = 32

    include_local_memory_overhead = True

    assert d % n == 0
    d_block = d //n
    k_blocks = n

    # various costs per client
    for r in range(rounds):
        # ===============================================================================================================
        # Intel-xeon-e5-2666-v3
        clock_rate = 3.2 * (10 ** 9)  # Assume we have CPU CLock 3.2 GHz
        number_of_cores = 10  # Number of cores
        hyperthreading = 2  # Hyperthreading
        ports_with_fpu_units = 2  # Number of FPU units
        mad_is_possible = True  # MAD is possible
        ports_with_load_store_units = 3  # Number of Load/Store Units per core

        peak_flops = clock_rate * number_of_cores * hyperthreading * ports_with_fpu_units
        if mad_is_possible:
            peak_flops = peak_flops * 2

        # Assumption: 1 closk per add or subtract, 1 clock per multiply, with throughput 1operation/clock.
        #   Approximately the true if look into characterstic of vectorized multiply add operations VFMADDPS in x86
        #   https://www.agner.org/optimize/instruction_tables.pdf , page 46

        addition_costs = [1.0 / peak_flops] * n  # Price in cloks converted to seconds
        mult_costs = [1.0 / peak_flops] * n      # Price in cloks converted to seconds

        # Number of samples
        samples_in_clients = [55 * mult_samples_num] +  [11 * mult_samples_num] * (n-1)

        # Substitute average across the globe: https://www.speedtest.net/global-index

        # bits/sec (MBps => bits/second)
        bandwith_c_to_m = 41.54 * (10 ** 6) / bandwithDivider  # uplink (client=>master)
        bandwith_m_to_c = 41.54 * (10 ** 6) / bandwithDivider  # downlink  (master=>client)

        rtts = [28 * (10 ** -3)] * n          # rtt time (milliseconds => seconds)

        # Turn of rtts
        #for i in range(len(rtts)): rtts[i] = 0.0
        # Make compute cheaper
        #for i in range(len(addition_costs)): addition_costs[i] /= 100000.0
        #for i in range(len(mult_costs)): mult_costs[i] /= 100000.0

        cache_line_size_in_bits = 64 * 8

        # Another memory access we currently ignore:
        #   L1 access 4 cycles
        l1_access = 4 / clock_rate / ports_with_load_store_units / number_of_cores / (cache_line_size_in_bits) * bits_per_component_value
        #   L2 access 10 cycles
        l2_access = 10 / clock_rate / ports_with_load_store_units / number_of_cores / (cache_line_size_in_bits) * bits_per_component_value
        #   L3 access 40 cycles
        l3_access = 40 / clock_rate / ports_with_load_store_units / number_of_cores / (cache_line_size_in_bits) * bits_per_component_value
        #   Main memory access depend on memory type installation, but for Intel Core series it's approximately:
        # https://www.intel.com/content/www/us/en/support/articles/000056722/processors/intel-core-processors.html
        ddr4_memory_access = 1.0 / (2933 * 2 * 8 * 4) / (cache_line_size_in_bits) * bits_per_component_value

        # ===============================================================================================================
        if include_local_memory_overhead:
            l1_access = 0.0
            l2_access = 0.0
            l3_access = 0.0
            ddr4_memory_access = 0.0
        # ===============================================================================================================
        mem_latency = l2_access * 2 # for two operands
        # ===============================================================================================================
        use_compression                      = True
        use_compression_coupled_with_compute = True
        reestimate_bandwith                  = True
        # ===============================================================================================================

        for i in range(n):
            addition_cost = addition_costs[i]
            mult_cost = mult_costs[i]
            samples_in_client = samples_in_clients[i]

            rtt = rtts[i]

            c = f"c{i + 1},k={r}"

            for ii in range(n):
                b = ii + 1
                g.addEdge(f"Compute g(xi) at {c} for b={b} [start]", f"Eval Ai@x at {c} for b={b} [start]", 0.0)
                #========================================================================================================
                if reestimate:
                    start_prev = times_prev[f"Eval Ai@x at {c} for b={b} [start]"]
                    end_prev = times_prev[f"Eval Ai@x - b at {c} [start]"]
                    assert end_prev >= start_prev

                    all_computes = [e for e in g_prev.E if "Eval Ai@x at {c} for b=" in e.source and f"{c}" not in e.source]
                    intersections = 0
                    for e in all_computes:
                        s_old = times_prev[e.source]
                        e_old = times_prev[e.source] + e.cost

                        if s_old > end_prev or e_old < start_prev:
                            continue
                        else:
                            start_intersection = max(s_old, start_prev)
                            end_intersection = min(e_old, end_prev)
                            intersections += (end_intersection - start_intersection) / (end_prev - start_prev)

                    multiplier = (k_blocks) / (intersections + 1)

                #========================================================================================================
                    g.addEdge(f"Eval Ai@x at {c} for b={b} [start]", f"Eval Ai@x - b at {c} [start]",
                              samples_in_client * (d_block - 1) * addition_cost * n / multiplier +
                              samples_in_client * d_block * mult_cost * n / multiplier +
                              samples_in_client * d_block * mem_latency + 1 * d_block * mem_latency)
                else:
                    g.addEdge(f"Eval Ai@x at {c} for b={b} [start]", f"Eval Ai@x - b at {c} [start]",
                              samples_in_client * (d_block - 1) * addition_cost * n +
                              samples_in_client * d_block * mult_cost * n +
                              samples_in_client * d_block * mem_latency + 1 * d_block * mem_latency)
                #========================================================================================================

            g.addEdge(f"Eval Ai@x - b at {c} [start]", f"Eval A^T(Ai@x - b) at {c} [start]", d * addition_cost + d * mem_latency)

            if use_compression:
                g.addEdge(f"Eval A^T(Ai@x - b) at {c} [start]", f"Scale A^T(Ai@x - b) at {c} with compression info [start]",
                           d_block * (samples_in_client - 1) * addition_cost + d_block * samples_in_client * mult_cost +
                           d_block * samples_in_client * mem_latency + 1 * samples_in_client * mem_latency)
                g.addEdge(f"Scale A^T(Ai@x - b) at {c} with compression info [start]", f"PermK compression at {c} [start]", d_block * mult_cost + d_block * mem_latency)
            else:
                g.addEdge(f"Eval A^T(Ai@x - b) at {c} [start]", f"Scale A^T(Ai@x - b) at {c} [start]",
                          d * (samples_in_client - 1) * addition_cost + d * samples_in_client * mult_cost +
                          d * samples_in_client * mem_latency + 1 * samples_in_client * mem_latency)
                g.addEdge(f"Scale A^T(Ai@x - b) at {c} [start]", f"PermK compression at {c} [start]",
                          d * mult_cost + d * mem_latency)

            g.addEdge(f"PermK compression at {c} [start]", f"AES Ecnryption at {c} [start]", d_block * mem_latency)

            g.addEdge(f"AES Ecnryption at {c} [start]", f"Send at {c} block {i+1}=> master[start]", d * mem_latency + d * addition_cost) # AESENC - 4/5 clocks for 128 bits input block => 1 clock per 32 bits.

            # Assume NIC read data via DMA directly from Main Memory
            if reestimate:
                start_prev = times_prev[f"Send at {c} block {i+1}=> master[start]"]
                end_prev   = times_prev[f"Sync at Master in k={r} for block {i+1}[start]"]
                assert end_prev >= start_prev

                all_sends_to_master = [e for e in g_prev.E if "=> master" in e.source and f"{c}" not in e.source]
                intersections = 0
                for e in all_sends_to_master:
                    s_old = times_prev[e.source]
                    e_old = times_prev[e.source] + e.cost

                    if s_old > end_prev or e_old < start_prev:
                        continue
                    else:
                        start_intersection = max(s_old, start_prev)
                        end_intersection   = min(e_old, end_prev)
                        intersections += (end_intersection - start_intersection) / (end_prev - start_prev)

                #multiplier = (k_blocks * n) / (intersections + 1)
                multiplier = (n) / (intersections + 1)


                g.addEdge(f"Send at {c} block {i+1}=> master[start]", f"Sync at Master in k={r} for block {i+1}[start]",
                          rtt / 2.0 + (d_block * bits_per_component_value) / (bandwith_c_to_m * multiplier) + d_block * l3_access)

            else:
                g.addEdge(f"Send at {c} block {i+1}=> master[start]", f"Sync at Master in k={r} for block {i+1}[start]",
                          rtt / 2.0 + (d_block * bits_per_component_value) / bandwith_c_to_m + d_block * l3_access)

        for i in range(n):
            b = i + 1
            g.addEdge(f"Sync at Master in k={r} for block {b}[start]", f"Store block at Master in k={r} for block {b} [start]", 0)
            g.addEdge(f"Store block at Master in k={r} for block {b} [start]", f"Ready to broadcast block at Master in k={r} for block {b} [start]", d_block*mem_latency)

        # Master just send to everybody step to apply
        for i in range(n):
            addition_cost = addition_costs[i]
            mult_cost = mult_costs[i]
            samples_in_client = samples_in_clients[i]
            rtt = rtts[i]

            c = f"c{i + 1},k={r}"
            cNext = f"c{i + 1},k={r + 1}"

            for i in range(n):
                b = i + 1
                g.addEdge( f"Ready to broadcast block at Master in k={r} for block {b} [start]", f"At {c} obtaining update, master=>{c} block {b} [start]",0)

                if reestimate:
                    start_prev = times_prev[f"At {c} obtaining update, master=>{c} block {b} [start]"]
                    end_prev = times_prev[f"At {c} make AES Decryption for block {b} [start]"]
                    assert end_prev >= start_prev

                    all_sends_from_master = [e for e in g_prev.E if "master=>" in e.source and f"{c}" not in e.source]

                    intersections = 0.0

                    for e in all_sends_from_master:
                        s_old = times_prev[e.source]
                        e_old = times_prev[e.source] + e.cost

                        if s_old > end_prev or e_old < start_prev:
                            continue
                        else:
                            start_intersection = max(s_old, start_prev)
                            end_intersection = min(e_old, end_prev)
                            intersections += (end_intersection - start_intersection) / (end_prev - start_prev)

                    # Totally there are only intersections + 1 from "n" which utilize communication bus
                    multiplier = k_blocks * n / (intersections + 1)
                    # print("===", multiplier)

                    g.addEdge(f"At {c} obtaining update, master=>{c} block {b} [start]", f"At {c} make AES Decryption for block {b} [start]",
                              rtt / 2.0 + (d_block * bits_per_component_value) / (bandwith_m_to_c*multiplier) + d_block * l3_access)

                else:
                    g.addEdge(f"At {c} obtaining update, master=>{c} block {b} [start]", f"At {c} make AES Decryption for block {b} [start]",
                              rtt / 2.0 + (d_block * bits_per_component_value) / (bandwith_m_to_c) + d_block * l3_access)

                g.addEdge(f"At {c} make AES Decryption for block {b} [start]", f"At {c} make a step for block {b} [start]", d * mem_latency + d * addition_cost)  # AESDEC - 4/5 clocks for 128 bits input block => 1 clock per 32 bits.

                g.addEdge(f"At {c} make a step for block {b} [start]", f"At {c} make a step for block {b} [end]",
                          d_block * addition_cost + 1.0 * d_block * mult_cost + d_block * mem_latency)  # step and scaling

                if r + 1 < rounds:
                    # Dependence for next iteration
                    g.addEdge(f"At {c} make a step for block {b} [end]", f"Compute g(xi) at {cNext} for b={i+1} [start]", 0)

    # Make parallel Job Scheduling
    res, times, dependecies = parallelJobScheduling(g)
    assert res == True

    return res, times, dependecies, g

if __name__ == "__main__":
    d = 10*1000*1000
    n = 4

    res, times_gd, dependecies_gd, g_gd = simulationGD(d, n)
    g_gd.dumpToDotWithCriticalPath("gd.txt", times_gd, dependecies_gd)
    timeLine(True, d, n, "GD", times_gd, dependecies_gd, g_gd)

    for i in range(10):
        res, times_gd_new, dependecies_gd_new, g_gd_new = simulationGD(d, n, True, times_gd, dependecies_gd, g_gd)
        g_gd_new.dumpToDotWithCriticalPath("gd_reschedule.txt", times_gd_new, dependecies_gd_new)
        timeLine(False, d, n, "GD RESCHEDULE", times_gd_new, dependecies_gd_new, g_gd_new)
        times_gd = times_gd_new
        dependecies_gd_new = dependecies_gd
        g_gd = g_gd_new

    timeLine(True, d, n, "GD REFINED", times_gd_new, dependecies_gd_new, g_gd_new)

    #==================================================================================================================
    bandwith_divider = 1
    # ==================================================================================================================
    res, times_pk, dependecies_pk, g_pk = simulationPermKSingleMaster(d, n, bandwith_divider)
    g_pk.dumpToDotWithCriticalPath("permk.txt", times_pk, dependecies_pk)

    timeLine(True, d, n, "PermK", times_pk, dependecies_pk, g_pk)

    for i in range(10):
        res, times_pk_new, dependecies_pk_new, g_pk_new = simulationPermKSingleMaster(d, n, bandwith_divider, True, times_pk, dependecies_pk, g_pk)
        g_pk_new.dumpToDotWithCriticalPath("permk_reschedule.txt", times_pk_new, dependecies_pk_new)
        timeLine(False, d, n, "PERMK REFINED", times_pk_new, dependecies_pk_new, g_pk_new)
        times_pk = times_pk_new
        dependecies_pk = dependecies_pk_new
        g_pk = g_pk_new

    timeLine(True, d, n, "PERMK REFINED", times_pk_new, dependecies_pk_new, g_pk_new)

    if not False:
        pass
        #timeLineForBlock(True, d, n, "PERMK/UPLINK REFINED b=1", times_pk_new, dependecies_pk_new, g_pk_new, 1, True, True, True)
        #timeLineForBlock(True, d, n, "PERMK/UPLINK REFINED b=2", times_pk_new, dependecies_pk_new, g_pk_new, 2, True, True, True)
        #timeLineForBlock(True, d, n, "PERMK/UPLINK REFINED b=3", times_pk_new, dependecies_pk_new, g_pk_new, 3, True, True, True)
        #timeLineForBlock(True, d, n, "PERMK/UPLINK REFINED b=4", times_pk_new, dependecies_pk_new, g_pk_new, 4, True, True, True)

        #timeLineForBlock(True, d, n, "PERMK/DOWNLINK REFINED b=1", times_pk_new, dependecies_pk_new, g_pk_new, 1, False, True, False)
        #timeLineForBlock(True, d, n, "PERMK/DOWNLINK REFINED b=2", times_pk_new, dependecies_pk_new, g_pk_new, 2, False, True, False)
        #timeLineForBlock(True, d, n, "PERMK/DOWNLINK REFINED b=3", times_pk_new, dependecies_pk_new, g_pk_new, 3, False, True, False)
        #timeLineForBlock(True, d, n, "PERMK/DOWNLINK REFINED b=4", times_pk_new, dependecies_pk_new, g_pk_new, 4, False, True, False)

        #timeLineForBlock(True, d, n, "PERMK/COMPUTE REFINED b=1", times_pk_new, dependecies_pk_new, g_pk_new, 1, False, False, True)
        #timeLineForBlock(True, d, n, "PERMK/COMPUTE REFINED b=2", times_pk_new, dependecies_pk_new, g_pk_new, 2, False, False, True)
        #timeLineForBlock(True, d, n, "PERMK/COMPUTE REFINED b=3", times_pk_new, dependecies_pk_new, g_pk_new, 3, False, False, True)
        #timeLineForBlock(True, d, n, "PERMK/COMPUTE REFINED b=4", times_pk_new, dependecies_pk_new, g_pk_new, 4, False, False, True)
    #uplink, downlink, computation):
    #==================================================================================================================

    #test_create_and_remove()
    #test_topological_sort_and_ac_sh_paths()
    #test_parallel_job_scheduling()
