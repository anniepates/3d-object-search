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

# if __name__ == "__main__": 
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