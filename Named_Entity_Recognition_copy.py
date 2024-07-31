# Keyword Extraction file using spaCy

import networkx as nx
import pandas as pd
import spacy
from spacy.tokens import Span
from spacy.matcher import PhraseMatcher
from collections import Counter
from string import punctuation

nlp = spacy.load("en_core_web_sm")
matcher = PhraseMatcher(nlp.vocab, attr="LOWER")

def create_dictionary(file_name): 
    df = pd.read_csv(file_name)
    content = {}
    for row in df.index:
        content[str(df['id'][row])] = df['content'][row]
    return content

def keyword_helper(content):
    kws = []
    doc = nlp(content)
    for word in doc:
        if (not(word.is_stop) and not (word.is_punct)):
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
    return matched  # The number of words each list has in common will become the edge weight

def create_kw_edges(summary_kws):
    edges = []
    used_ids = []
    lexicon = set()  # TODO: deleteme ?
    for entry in summary_kws:
        for word in summary_kws[entry]:
            lexicon.add(word)

    for id1 in summary_kws.keys():
        used_ids.append(id1)
        for id2 in summary_kws.keys():
            if id2 not in used_ids:
                matched = match_keywords(summary_kws[id1], summary_kws[id2])
                weight = len(matched) / ((len(summary_kws[id1]) + len(summary_kws[id2]))/2)  # TODO: replace with TF_IDF? word2vec?
                edges.append((id1, id2, weight, matched))
    print("Edges: ", edges)
    return edges

def build_kw_graph(kws):
    # print("summaries: ", summaries)
    # kws = keyword_summaries(summaries) 
    print("kws: ", kws)
    edges_v = create_kw_edges(kws)
    G = nx.MultiGraph()
    for key in kws:
        G.add_node(key, kws=kws[key])
    for edge in edges_v:
        G.add_edge(edge[0], edge[1], weight=edge[2], common=edge[3])

    print("Nodes: ", list(G.nodes))
    print("Created a graph with ", len(list(G.nodes)), " nodes and ", len(list(G.edges)), "edges") # % (len(list(G.nodes)), len(list(G.edges)))
    return G

"""
Get segment similarities from a csv matrix with segment id's on row & column labels
"""
def load_seg_similaritites(file_name):
    df = pd.read_csv(file_name, header=0, index_col=0)
    return df

# def get_seg_similarity(df, seg1, seg2):
#     return(df.at[seg1, seg2])

"""
Creates multiedge graph of mesh models
Edges connect each model containing each segment from each model and their similarity
Ex. If Model A has a cup & handle segment, and Model B has a cup, edges will look something like
    [modelA, modelB, {(mAcup, mBcup): .9}], [modelA, modelB, {(mAhandle, mBcup): .2}]
"""
def create_seg_edges(seg_ids, file_name):  # TODO: kws here is really segment ids
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
    # print("summaries: ", summaries)
    # kws = keyword_summaries(summaries) 
    print("kws: ", kws)
    edges_v = create_seg_edges(kws, file_name)
    G = nx.MultiGraph()
    for key in kws:
        G.add_node(key, segs=kws[key])
    for edge in edges_v:
        G.add_edge(edge[0], edge[1], weight=edge[2], segs=edge[3])

    print("Nodes: ", list(G.nodes))
    print("Created a graph with ", len(list(G.nodes)), " nodes and ", len(list(G.edges)), "edges") # % (len(list(G.nodes)), len(list(G.edges)))
    return G

def update_probabilities(seg_ids : dict[tuple[str, str], float], seg_similarity_file, desc_kws):
    g_model = build_seg_graph(seg_ids, seg_similarity_file)
    g_desc = build_kw_graph(desc_kws)
    probs = {}
    ev : dict[tuple[str, str], float] = {}
    relevant_kws = {}

    # Initialize some probabilities
    for model_node, model_node_attrs in g_model.nodes.data():
            for seg in model_node_attrs["segs"]:
                # print(g_desc.nodes[model_node]["kws"])
                # Find corresponding description and its keywords
                for kw in g_desc.nodes[model_node]["kws"]:
                    # Set initial probability to suggest segment could relate to any of the relevant kws
                    new_prob = 1.0   #/float(len(g_desc.nodes[model_node]["kws"]))
                    if ((seg, kw) in probs):
                        probs.update({(seg, kw): new_prob})
                    else:
                        probs[(seg, kw)] = new_prob                    
                    # Initialize expected value of probability at 1    
                    ev[(seg, kw)] = 1
                    # Keep track of relevant keywords for the segment
                    if seg in relevant_kws.keys():
                        relevant_kws[seg].add(kw)
                    else:
                        relevant_kws[seg] = {kw}


    print("INITIAL PROBABILITIES: ", probs)
    # print("INITIAL LABELS: ", relevant_kws)

    # Update expected values using segment similarity info
    for edge in g_model.edges:
        model, neighbor, ind = edge[0], edge[1], edge[2]
        model_seg = g_model.edges[model, neighbor, ind]["segs"][0]
        neighbor_seg = g_model.edges[model, neighbor, ind]["segs"][1]
        weight = g_model.edges[model, neighbor, ind]["weight"]  # TODO: rename to seg_similarity
        print(edge, g_model.edges[model, neighbor, ind]["segs"])

        for neighbor_kw in g_desc.nodes[neighbor]["kws"]:
            if (neighbor_seg, neighbor_kw) in probs:
                node_new_val = probs[(neighbor_seg, neighbor_kw)] * weight
            else:  # TODO: make an assert - we shouldn't hit this
                node_new_val = 1.0/float(len(g_desc.nodes[neighbor]['kws'])) * weight    
            if (model_seg, neighbor_kw) not in ev: 
                ev[(model_seg, neighbor_kw)] = .5  # Possible the model segment won't be associated with the neighbor keyword       
            ev.update({(model_seg, neighbor_kw): ev[(model_seg, neighbor_kw)] + node_new_val})
            # Update expected values and relevant keywords if new ones are discovered
            if neighbor_kw not in relevant_kws[model_seg]: 
                relevant_kws[model_seg].add(neighbor_kw)
            print("Model: ", model_seg, neighbor_kw, ev[(model_seg, neighbor_kw)])

        # Update the ev's for the neighbor too TODO: make this a function
        for keyword in g_desc.nodes[model]["kws"]:
            if (model_seg, keyword) in probs:
                neigh_new_val = probs[(model_seg, keyword)] * weight
            else:
                neigh_new_val = 1.0/float(len(g_desc.nodes[model]['kws'])) * weight
            if (neighbor_seg, keyword) not in ev: 
                ev[(neighbor_seg, keyword)] = .5  # Possible the neighbor segment won't be associated with the keyword
            ev.update({(neighbor_seg, keyword): 
                       ev[(neighbor_seg, keyword)] + neigh_new_val})
            # Add kw to list of possible words for neighbor segment
            if keyword not in relevant_kws[neighbor_seg]: relevant_kws[neighbor_seg].add(keyword)
            print("Neighbor: ", neighbor_seg, keyword, ev[(neighbor_seg, keyword)])        
        # break

    # # Update intermediate values using segment similarity info
    # for model_node, model_node_attrs in g_model.nodes.data():
    #     # for desc_node, desc_node_attrs in g_desc.nodes.data():
    #         for seg in model_node_attrs["segs"]:
    #             for kw in g_desc.nodes[model_node]["kws"]:
    #                 for neighbor_node_m in g_model.neighbors(model_node):
    #                     # for neighbor_node_d in g_desc.neighbors(desc_node):
    #                         for neighbor_seg in g_model.nodes[neighbor_node_m]["segs"]:
    #                             for neighbor_kw in g_desc.nodes[neighbor_node_m]["kws"]:
    #                                 for i in range(g_model.number_of_edges(model_node, neighbor_node_m)):
    #                                     if g_model.edges[model_node, neighbor_node_m, i]["segs"] == (seg, neighbor_seg):
    #                                         # print("edge: ", g_model.edges[model_node, neighbor_node_m, i])
    #                                         # Update the expected value by multiplying initial value by the similarity of the segments
    #                                         node_new_val = probs[(seg, kw)] * g_model.edges[model_node, neighbor_node_m, i]['weight']
    #                                         # print("weight: ", g_model.edges[model_node, neighbor_node_m, i]['weight'])
    #                                         # print("Node kw: ", kw)
    #                                         # print("Neighbor kw: ", neighbor_kw)
    #                                         # print("Node new val: ", seg, node_new_val)
    #                                         if (neighbor_seg, kw) in probs:
    #                                             neigh_new_val = probs[(neighbor_seg, kw)] * g_model.edges[model_node, neighbor_node_m, i]['weight']
    #                                         else:
    #                                             neigh_new_val = .25 * g_model.edges[model_node, neighbor_node_m, i]['weight']
    #                                         # print("Neighbor new val: ", neighbor_seg, neigh_new_val)

    #                                         # Update expected values and relevant keywords if new ones are discovered
    #                                         old_val = ev[(seg, kw)] if ev[(seg, kw)] else 1
    #                                         if (neighbor_seg, kw) not in ev: ev[(neighbor_seg, kw)] = 1
    #                                         ev.update({(seg, kw): ev[(neighbor_seg, kw)] + node_new_val})
    #                                         if kw not in relevant_kws[seg]: relevant_kws[seg].add(kw)
    #                                         ev.update({(neighbor_seg, neighbor_kw): old_val + neigh_new_val})
    #                                         if neighbor_kw not in relevant_kws[neighbor_seg]: relevant_kws[seg].add(kw)
    #                                         print('\n', i, seg, neighbor_seg, kw, "Intermediate vals:", ev)
    #                             # break
    #                         # break
    #                     # break
    #                 # break
    #             # break
    #         # break


    # Update probabilities using intermediary values
    for model_node, model_node_attrs in g_model.nodes.data():
        # for desc_node, desc_node_attrs in g_desc.nodes.data():
        for seg in relevant_kws:
            for label in relevant_kws[seg]:
                probs[(seg, label)] = ev[(seg, label)] / len(relevant_kws[seg])

    # print("UPDATED VALUES: ", ev)
    print("UPDATED PROBABILITIES: ", probs)
                    
    return g_model, g_desc


# Old methods to break a semantic edge attribute and see which segment edge attr was affected
# TODO: deleteme
def break_edge(kws, word_to_remove):
    for key in kws:
        if word_to_remove in kws[key]:
            kws[key].remove(word_to_remove)
    return kws

def find_corresponding_segment(desc_graph, broken_graph, segment_graph):
    edges = desc_graph.edges
    desc_weights = nx.get_edge_attributes(desc_graph, "weight")
    broken_weights = nx.get_edge_attributes(broken_graph, "weight")
    segment_attrs = nx.get_edge_attributes(segment_graph, "common")
    candidates = set()
    non_candidates = set()
    matching_seg = set()

    for edge in edges:
        if (desc_weights[edge] != broken_weights[edge]):
            for seg in segment_attrs[edge]:
                candidates.add(seg)
        else:
            for seg in segment_attrs[edge]:
                non_candidates.add(seg)

    for candidate in candidates:
        if candidate not in non_candidates:
            matching_seg.add(candidate)

    return matching_seg



"""
keywords = ["pill", "pill box", "pillbox", 
            "prosthetic", "prosthesis", 
            "disabled", "disability", "disabilities", "impaired", "impair", "handicap", "handicapped", 
            "visual", "visually impaired", "visual impairment",
            "enable",
            "tactile", "tactile grpahics",
            "assistive device", "assistive", "assistive technology",
            "Braille", 
            "grip",
            "wheelchair",
            "access", "accessibility",
            "amputee",
            "medicine",
            "elderly", "old", "senior", "aged",
            "cane"]
patterns = [nlp.make_doc(text) for text in keywords]
matcher.add("TerminologyList", patterns)

descriptions = ["This 3d printable prosthetic finger ideal for those missing two finger segments, though it can be configured for either 2 or 1 knuckles. Open it with the configurator to enter your own values and generate a custom model. Many options are configurable, so with the right measurements and tweaking it should be adaptable to most people's needs - though it takes some trial and error to get a perfect fit. See the measurement guide PDF for instructions on measuring and configuring the model.",
                "A pillbox with eight compartments. One for each weekday and one to be selected during transport. The lid snaps in permanently (you might have to press tightly to connect both parts). A locking mechanism prevents unwanted opening of the compartments. To rotate the lid, pull it gently while turning.",
                "Pillbox, round, pills for one month .",
                "Prothestic Hand\nRemix: http://www.thingiverse.com/thing:1095104\nIn this video explains the process. Enables subtitles\nhttps://youtu.be/uWL13vvi94s\nThis prosthesis has been designed to help those people with missing fingers.\nIt has been modelled keeping in mind the following principles:\n-Keep the number of printed parts as low as possible.\n-Keep the assembly process as simple as possible.\n-Use the least possible amount of non-printable parts. In this new model, you can choose how to join fingers, using screws (2M), axes wire, or filament printer.\nLeave openings in the model to allow the hand to transpire.\n-The previous model had another hand rotation system on the wrist, had more freedom. On this occasion I have modified the movement of the wrist using the fantastic idea of a union that uses hand 'Flexy-hand' http://www.thingiverse.com/thing:380665\n-I have also modified the system of tensioning the strings. It is safer than before and takes up less space.\nThe fingers flex when they are pulled by strings, such as fishing line, and extend automatically with the help of elastic cord. A fabric strap with velcro is needed to secure the prosthesis to the arm. You can download the original file blender, and thus can be studied and modified to improve its performance."]


def print_keywords(description_list):
        for i in range(len(description_list)):
            doc = nlp(description_list[i])
            matches = matcher(doc)
            #print(matches)
            for match_id, start, end in matches:
                span = doc[start:end]
                print(span.text)   

def extract_POS(sample_doc):
    res = []
    for chk in sample_doc.noun_chunks:
        tmp = ""
        for tkn in chk:
            if tkn.pos_ in ['NOUN', 'PROPN', 'ADJ']:
                if (not(tkn.is_stop) and not (tkn.is_punct)):
                    tmp = tmp + tkn.text.lower() + " "
        if (tmp.strip()!=""):
           res.append(tmp.strip())
    return list(dict.fromkeys(res))
"""


if __name__ == "__main__": 
    # Print keywords found on keyword list
    #print_keywords(descriptions)
    # Note this doesn't account  for spelling error in 4th description "Prothestic"

    # Extract nouns, proper nouns, and adjectives from description
    # description_pos = extract_POS(nlp(descriptions[0]))
    # print(description_pos)
    # query = "Prosthetic finger for someone missing two finger segments. Gives a perfect fit"
    # query_POS = extract_POS(nlp(query))
    # print(query_POS)
    # matched, unmatched = matching_keywords(query_POS, description_pos)
    # print("Matching terms: ", matched)
    # #percentage of matched keywords
    # print("matching percentage using unique words:", f"{len(matched)/len(query_POS):.00%}")
    # summaries = create_summary_dictionary("summaries.csv")
    # kws = keyword_summaries(summaries)   
    # # print(kws['3519963'])
    # # print(kws['1506985'])
    # # print(match_keywords(kws['3519963'], kws['1506985']))
    # edges = creating_edges(kws)
    # print(edges)


    descriptions = create_dictionary("one_descriptions.csv")
    desc_kws = keyword_summaries(descriptions)
    # description_graph = build_kw_graph(desc_kws)
    # description_graph.add_nodes_from(desc_kws.keys)
    # description_graph.add_edges_from()
    # print("Description graph: ", description_graph.edges)
    # print("Desc edge attrs: ", nx.get_edge_attributes(description_graph, "common"), "\n")
    segments = create_dictionary("one_segments.csv")
    # print(get_seg_similarity("groundtruth_seg_similarities.csv", 'm1tube', 'm2tube'))
    seg_kws = keyword_summaries(segments)
    # segment_graph = build_seg_graph(seg_kws, "groundtruth_seg_similarities.csv")
    # print("Seg edge attrs: ", nx.get_edge_attributes(segment_graph, "segs"), "\n")

    update_probabilities(seg_kws, "one_seg_similarities.csv", desc_kws)


    # TODO: deleteme broken edges
    # broken_desc_kws = break_edge(desc_kws, 'handle')
    # broken_desc_graph = build_graph(broken_desc_kws)
    # print("Broken edge attrs: ", nx.get_edge_attributes(broken_desc_graph, "common"))
    # answer = find_corresponding_segment(description_graph, broken_desc_graph, segment_graph)
    # print(answer)
