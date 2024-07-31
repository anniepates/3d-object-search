"""
Main file to run algorithm.
Takes 3 files: #TODO: take as keyboard params
Files can be swapped so long as they correspond to each other (ie the first word is the same)
"""

import kw_extraction as kwe
import build_graph as gr

if __name__ == "__main__":
    desc_data = "graph_data/one_descriptions.csv"
    seg_data = "graph_data/one_segments.csv"
    similarities_data = "graph_data/one_seg_similarities.csv"

    descriptions = kwe.create_dictionary(desc_data)
    desc_kws = gr.keyword_summaries(descriptions)
    # description_graph = build_kw_graph(desc_kws)
    # description_graph.add_nodes_from(desc_kws.keys)
    # description_graph.add_edges_from()
    # print("Description graph: ", description_graph.edges)
    # print("Desc edge attrs: ", nx.get_edge_attributes(description_graph, "common"), "\n")
    segments = gr.create_dictionary(seg_data)
    # print(get_seg_similarity("groundtruth_seg_similarities.csv", 'm1tube', 'm2tube'))
    seg_kws = gr.keyword_summaries(segments)
    # segment_graph = build_seg_graph(seg_kws, "groundtruth_seg_similarities.csv")
    # print("Seg edge attrs: ", nx.get_edge_attributes(segment_graph, "segs"), "\n")

    gr.update_probabilities(seg_kws, similarities_data, desc_kws)