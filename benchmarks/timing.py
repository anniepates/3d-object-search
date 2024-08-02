import networkx as nx
import pandas as pd
import time
import memory_profiler
import cProfile
import pstats
import build_graph

# Decorator to measure execution time
def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.4f} seconds")
        return result
    return wrapper

@timing_decorator
def create_dictionary(file_name):
    df = pd.read_csv(file_name)
    content = {}
    for row in df.index:
        content[str(df['id'][row])] = df['content'][row]
    return content

@timing_decorator
def keyword_helper(content):
    kws = []
    doc = build_graph.nlp(content)
    for word in doc:
        if (not(word.is_stop) and not (word.is_punct)):
            tmp = word.text.strip()
            if tmp != "":
                kws.append(tmp.lower())
    return kws

@timing_decorator
def create_kw_edges(summary_kws):
    edges = []
    used_ids = []
    lexicon = set()
    for entry in summary_kws:
        for word in summary_kws[entry]:
            lexicon.add(word)

    for id1 in summary_kws.keys():
        used_ids.append(id1)
        for id2 in summary_kws.keys():
            if id2 not in used_ids:
                matched = build_graph.match_keywords(summary_kws[id1], summary_kws[id2])
                weight = len(matched) / ((len(summary_kws[id1]) + len(summary_kws[id2]))/2)
                edges.append((id1, id2, weight, matched))
    print("Edges: ", edges)
    return edges

@timing_decorator
def build_kw_graph(kws):
    edges_v = create_kw_edges(kws)
    G = nx.MultiGraph()
    for key in kws:
        G.add_node(key, kws=kws[key])
    for edge in edges_v:
        G.add_edge(edge[0], edge[1], weight=edge[2], common=edge[3])

    print("Nodes: ", list(G.nodes))
    print("Created a graph with ", len(list(G.nodes)), " nodes and ", len(list(G.edges)), "edges")
    return G

@timing_decorator
def update_probabilities(seg_ids, seg_similarity_file, desc_kws):
    g_model = build_graph.build_seg_graph(seg_ids, seg_similarity_file)
    g_desc = build_kw_graph(desc_kws)
    probs = {}
    ev = {}
    relevant_kws = {}

    # Initialize some probabilities
    for model_node, model_node_attrs in g_model.nodes.data():
        for seg in model_node_attrs["segs"]:
            for kw in g_desc.nodes[model_node]["kws"]:
                new_prob = 1.0
                if ((seg, kw) in probs):
                    probs.update({(seg, kw): new_prob})
                else:
                    probs[(seg, kw)] = new_prob
                ev[(seg, kw)] = 1
                if seg in relevant_kws.keys():
                    relevant_kws[seg].add(kw)
                else:
                    relevant_kws[seg] = {kw}

    print("INITIAL PROBABILITIES: ", probs)

    # Update expected values using segment similarity info
    for edge in g_model.edges:
        model, neighbor, ind = edge[0], edge[1], edge[2]
        model_seg = g_model.edges[model, neighbor, ind]["segs"][0]
        neighbor_seg = g_model.edges[model, neighbor, ind]["segs"][1]
        weight = g_model.edges[model, neighbor, ind]["weight"]
        print(edge, g_model.edges[model, neighbor, ind]["segs"])

        for neighbor_kw in g_desc.nodes[neighbor]["kws"]:
            if (neighbor_seg, neighbor_kw) in probs:
                node_new_val = probs[(neighbor_seg, neighbor_kw)] * weight
            else:
                node_new_val = 1.0/float(len(g_desc.nodes[neighbor]['kws'])) * weight
            if (model_seg, neighbor_kw) not in ev:
                ev[(model_seg, neighbor_kw)] = .5
            ev.update({(model_seg, neighbor_kw): ev[(model_seg, neighbor_kw)] + node_new_val})
            if neighbor_kw not in relevant_kws[model_seg]:
                relevant_kws[model_seg].add(neighbor_kw)
            print("Model: ", model_seg, neighbor_kw, ev[(model_seg, neighbor_kw)])

        for keyword in g_desc.nodes[model]["kws"]:
            if (model_seg, keyword) in probs:
                neigh_new_val = probs[(model_seg, keyword)] * weight
            else:
                neigh_new_val = 1.0/float(len(g_desc.nodes[model]['kws'])) * weight
            if (neighbor_seg, keyword) not in ev:
                ev[(neighbor_seg, keyword)] = .5
            ev.update({(neighbor_seg, keyword): ev[(neighbor_seg, keyword)] + neigh_new_val})
            if keyword not in relevant_kws[neighbor_seg]: relevant_kws[seg].add(keyword)
            print("Neighbor: ", neighbor_seg, keyword, ev[(neighbor_seg, keyword)])

    for model_node, model_node_attrs in g_model.nodes.data():
        for seg in relevant_kws:
            for label in relevant_kws[seg]:
                probs[(seg, label)] = ev[(seg, label)] / len(relevant_kws[seg])

    print("UPDATED PROBABILITIES: ", probs)
    return g_model, g_desc

# Memory profiling
@memory_profiler.profile
def run_benchmark(file_name, seg_similarity_file, desc_kws):
    summaries = create_dictionary(file_name)
    kws = build_graph.keyword_summaries(summaries)
    g_kw = build_kw_graph(kws)
    seg_ids = create_dictionary(seg_similarity_file)
    update_probabilities(seg_ids, seg_similarity_file, desc_kws)

if __name__ == "__main__":
    file_name = 'your_file.csv'
    seg_similarity_file = 'your_seg_similarity_file.csv'
    desc_kws = {
        'desc1': ['keyword1', 'keyword2'],
        'desc2': ['keyword3', 'keyword4']
    }

    # Profiling
    profiler = cProfile.Profile()
    profiler.enable()
    run_benchmark(file_name, seg_similarity_file, desc_kws)
    profiler.disable()

    stats = pstats.Stats(profiler)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.print_stats()
