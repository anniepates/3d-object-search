import networkx as nx
import pandas as pd
import spacy
from spacy.tokens import Span
from spacy.matcher import PhraseMatcher
from collections import Counter
from string import punctuation
from time import time
import memory_profiler
import cProfile
import pstats
import io
import random
import string

nlp = spacy.load("en_core_web_sm")
matcher = PhraseMatcher(nlp.vocab, attr="LOWER")


def create_dictionary(file_name):
    df = pd.read_csv(file_name)
    content = {}
    for row in df.index:
        content[str(df['id'][row])] = df['content'][row] if pd.notna(df['content'][row]) else ""
    return content


def keyword_helper(content):
    kws = []
    doc = nlp(content)
    for word in doc:
        if not word.is_stop and not word.is_punct:
            tmp = word.text.strip()
            if tmp != "":
                kws.append(tmp.lower())
    return kws


def keyword_summaries(summaries):
    return dict((k, keyword_helper(v)) for k, v in summaries.items())


def match_keywords(query_list, entry_list):
    matched = []
    unmatched = []

    for x in query_list:
        if x in entry_list:
            matched.append(x)
        else:
            unmatched.append(x)
    return matched


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
                matched = match_keywords(summary_kws[id1], summary_kws[id2])
                weight = len(matched) / ((len(summary_kws[id1]) + len(summary_kws[id2])) / 2)
                edges.append((id1, id2, weight, matched))
    print("Edges: ", edges)
    return edges


def build_kw_graph(kws):
    print("kws: ", kws)
    edges_v = create_kw_edges(kws)
    G = nx.MultiGraph()
    for key in kws:
        G.add_node(key, kws=kws[key])
    for edge in edges_v:
        G.add_edge(edge[0], edge[1], weight=edge[2], common=edge[3])

    print("Nodes: ", list(G.nodes))
    print("Created a graph with ", len(list(G.nodes)), " nodes and ", len(list(G.edges)), "edges")
    return G


def load_seg_similaritites(file_name):
    df = pd.read_csv(file_name, header=0, index_col=0)
    return df


def create_seg_edges(seg_ids, file_name):
    df = pd.read_csv(file_name, header=0, index_col=0)

    edges = []
    used_ids = []

    for id1 in seg_ids.keys():
        used_ids.append(id1)
        for id2 in seg_ids.keys():
            if id2 not in used_ids:
                for id1seg in seg_ids[id1]:
                    for id2seg in seg_ids[id2]:
                        weight = df.at[id1seg, id2seg]
                        edges.append((id1, id2, weight, (id1seg, id2seg)))
    print("Edges: ", edges)
    return edges


def build_seg_graph(kws, file_name):
    print("kws: ", kws)
    edges_v = create_seg_edges(kws, file_name)
    G = nx.MultiGraph()
    for key in kws:
        G.add_node(key, segs=kws[key])
    for edge in edges_v:
        G.add_edge(edge[0], edge[1], weight=edge[2], segs=edge[3])

    print("Nodes: ", list(G.nodes))
    print("Created a graph with ", len(list(G.nodes)), " nodes and ", len(list(G.edges)), "edges")
    return G


def update_probabilities(seg_ids, seg_similarity_file, desc_kws):
    g_model = build_seg_graph(seg_ids, seg_similarity_file)
    g_desc = build_kw_graph(desc_kws)
    probs = {}
    ev = {}
    relevant_kws = {}

    for model_node, model_node_attrs in g_model.nodes.data():
        for seg in model_node_attrs["segs"]:
            for kw in g_desc.nodes[model_node]["kws"]:
                new_prob = 1.0
                probs[(seg, kw)] = new_prob
                ev[(seg, kw)] = 1
                if seg in relevant_kws.keys():
                    relevant_kws[seg].add(kw)
                else:
                    relevant_kws[seg] = {kw}

    print("INITIAL PROBABILITIES: ", probs)

    for edge in g_model.edges:
        model, neighbor, ind = edge[0], edge[1], edge[2]
        model_seg = g_model.edges[model, neighbor, ind]["segs"][0]
        neighbor_seg = g_model.edges[model, neighbor, ind]["segs"][1]
        weight = g_model.edges[model, neighbor, ind]["weight"]

        for neighbor_kw in g_desc.nodes[neighbor]["kws"]:
            node_new_val = probs[(neighbor_seg, neighbor_kw)] * weight if (neighbor_seg,
                                                                           neighbor_kw) in probs else 1.0 / float(
                len(g_desc.nodes[neighbor]['kws'])) * weight
            ev[(model_seg, neighbor_kw)] = ev[(model_seg, neighbor_kw)] + node_new_val if (model_seg,
                                                                                           neighbor_kw) in ev else .5
            relevant_kws[model_seg].add(neighbor_kw) if neighbor_kw not in relevant_kws[model_seg] else None
            print("Model: ", model_seg, neighbor_kw, ev[(model_seg, neighbor_kw)])

        for keyword in g_desc.nodes[model]["kws"]:
            neigh_new_val = probs[(model_seg, keyword)] * weight if (model_seg, keyword) in probs else 1.0 / float(
                len(g_desc.nodes[model]['kws'])) * weight
            ev[(neighbor_seg, keyword)] = ev[(neighbor_seg, keyword)] + neigh_new_val if (neighbor_seg,
                                                                                          keyword) in ev else .5
            relevant_kws[neighbor_seg].add(keyword) if keyword not in relevant_kws[neighbor_seg] else None
            print("Neighbor: ", neighbor_seg, keyword, ev[(neighbor_seg, keyword)])

    for model_node, model_node_attrs in g_model.nodes.data():
        for seg in relevant_kws:
            for label in relevant_kws[seg]:
                probs[(seg, label)] = ev[(seg, label)] / len(relevant_kws[seg])

    print("UPDATED PROBABILITIES: ", probs)

    return g_model, g_desc


def profile_memory_and_cpu(file_name, size_label):
    profiler = cProfile.Profile()
    profiler.enable()

    m_prof = memory_profiler.memory_usage(proc=-1, interval=0.1, timeout=1)

    summaries = create_dictionary(file_name)
    kws = keyword_summaries(summaries)
    build_kw_graph(kws)

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')

    # Save profile data to string buffer
    s = io.StringIO()
    stats = pstats.Stats(profiler, stream=s).sort_stats('cumtime')
    stats.print_stats()
    profile_data = s.getvalue()

    print(f"Profile for {size_label} data:\n", profile_data)
    print(f'Memory usage for {size_label} data: {m_prof} MB')

    return profile_data, m_prof


def generate_example_data():
    sizes = {
        "small": 10,
        "medium": 100,
        "large": 1000
    }
    for size_label, size in sizes.items():
        data = {
            "id": range(size),
            "content": [" ".join(random.choices(string.ascii_lowercase + " ", k=100)) for _ in range(size)]
        }
        df = pd.DataFrame(data)
        df.to_csv(f"{size_label}_data.csv", index=False)
        print(f"{size_label.capitalize()} dataset created with {size} entries.")


def run_profiling():
    generate_example_data()

    datasets = {
        "small": "small_data.csv",
        "medium": "medium_data.csv",
        "large": "large_data.csv"
    }

    results = {}

    for size_label, file_name in datasets.items():
        start_time = time()
        profile_data, memory_usage = profile_memory_and_cpu(file_name, size_label)
        end_time = time()
        elapsed_time = end_time - start_time
        results[size_label] = {
            "time": elapsed_time,
            "profile_data": profile_data,
            "memory_usage": memory_usage
        }
        print(f"Time taken for {size_label} data: {elapsed_time} seconds\n")

    # Draw conclusions
    for size_label, result in results.items():
        print(f"\n--- {size_label.capitalize()} Data ---")
        print(f"Time taken: {result['time']} seconds")
        print(f"Memory usage: {result['memory_usage']} MB")
        print(f"CPU Profile:\n{result['profile_data']}")


if __name__ == "__main__":
    run_profiling()
