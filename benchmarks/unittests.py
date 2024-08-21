import unittest
from unittest.mock import patch, mock_open
import pandas as pd
from io import StringIO
import networkx as nx
import spacy
from build_graph import create_dictionary, keyword_helper, keyword_summaries, match_keywords, create_kw_edges, \
    build_kw_graph, load_seg_similaritites, create_seg_edges, build_seg_graph, update_probabilities

nlp = spacy.load("en_core_web_sm")


class TestKeywordExtraction(unittest.TestCase):

    def setUp(self):
        self.sample_csv = "id,content\n1,This is a test.\n2,Another test.\n3,\n4,Special chars !@# and punctuations."
        self.empty_csv = "id,content\n"
        self.malformed_csv = "id,content\n1,This is a test\n2 Another test"

        self.valid_summaries = {
            "1": "This is a test summary.",
            "2": "Another summary here."
        }
        self.empty_summaries = {}
        self.large_summaries = {
            "1": "This is a very long test string. It contains many different words, "
                 "including some that are not stop words. The purpose of this test is to "
                 "ensure that the keyword extraction works correctly even with large "
                 "inputs.",
            "2": "This is another long test string. It also contains many different "
                 "words, and its purpose is to test the robustness of the keyword "
                 "extraction function.",
            "3": "This is yet another example of a long test string. The keyword "
                 "extraction function should be able to handle it without any issues.",
            "4": "Short string.",
            "5": "A string with numbers 12345 and special characters !@#.",
            "6": "A string with mixed-case WORDS and stop words and punctuations."
        }

        self.valid_seg_ids = {
            "1": ["seg1", "seg2"],
            "2": ["seg2", "seg3"]
        }
        self.empty_seg_ids = {}
        self.large_seg_ids = {
            "1": ["seg1", "seg2", "seg3", "seg4"],
            "2": ["seg2", "seg3", "seg4", "seg5"],
            "3": ["seg3", "seg4", "seg5", "seg1"],
            "4": ["seg1", "seg3", "seg5", "seg2"]
        }

        self.seg_similarity_csv = "seg1,seg2,seg3,seg4,seg5\nseg1,1,0.5,0.2,0.3,0.1\nseg2,0.5,1,0.3,0.4,0.2\nseg3,0.2,0.3,1,0.5,0.3\nseg4,0.3,0.4,0.5,1,0.6\nseg5,0.1,0.2,0.3,0.6,1"
        self.invalid_seg_similarity_csv = "seg1,seg2,seg3\nseg1,1,0.5\nseg2,0.5,1,0.3"

        self.desc_kws = {
            "1": ["test", "summary"],
            "2": ["another", "summary"]
        }
        self.large_desc_kws = {
            "1": ["keyword1", "keyword2", "keyword3"],
            "2": ["keyword2", "keyword3", "keyword4"],
            "3": ["keyword3", "keyword4", "keyword5"],
            "4": ["keyword1", "keyword3", "keyword5"]
        }

    def test_create_dictionary_valid(self):
        with patch("builtins.open", mock_open(read_data=self.sample_csv)):
            with patch("pandas.read_csv", return_value=pd.read_csv(StringIO(self.sample_csv))):
                result = create_dictionary("fake_file.csv")
                expected = {"1": "This is a test.", "2": "Another test.", "3": "",
                            "4": "Special chars !@# and punctuations."}
                self.assertEqual(result, expected)

    def test_create_dictionary_empty(self):
        with patch("builtins.open", mock_open(read_data=self.empty_csv)):
            with patch("pandas.read_csv", return_value=pd.read_csv(StringIO(self.empty_csv))):
                result = create_dictionary("fake_file.csv")
                expected = {}
                self.assertEqual(result, expected)

    def test_create_dictionary_malformed(self):
        with self.assertRaises(pd.errors.ParserError):
            with patch("builtins.open", mock_open(read_data=self.malformed_csv)):
                with patch("pandas.read_csv", side_effect=pd.errors.ParserError):
                    create_dictionary("fake_file.csv")

    def test_keyword_helper_normal(self):
        result = keyword_helper("This is a test string for keyword extraction.")
        expected = ["test", "string", "keyword", "extraction"]
        self.assertEqual(result, expected)

    def test_keyword_helper_empty(self):
        result = keyword_helper("")
        expected = []
        self.assertEqual(result, expected)

    def test_keyword_helper_stop_words(self):
        result = keyword_helper("and, or, but, the, ...")
        expected = []
        self.assertEqual(result, expected)

    def test_keyword_helper_large(self):
        large_text = ("This is a large text with a lot of different words. It should include many different "
                      "words and ensure that the keyword extraction can handle a large variety of input without "
                      "any issues, including multiple sentences and various punctuation marks!")
        result = keyword_helper(large_text)
        expected = ["large", "text", "lot", "different", "words", "include", "different", "words", "ensure",
                    "keyword", "extraction", "handle", "large", "variety", "input", "issues", "including",
                    "multiple", "sentences", "punctuation", "marks"]
        self.assertEqual(result, expected)

    def test_keyword_summaries_valid(self):
        result = keyword_summaries(self.valid_summaries)
        expected = {
            "1": ["test", "summary"],
            "2": ["summary"]
        }
        self.assertEqual(result, expected)

    def test_keyword_summaries_empty(self):
        result = keyword_summaries(self.empty_summaries)
        expected = {}
        self.assertEqual(result, expected)

    def test_keyword_summaries_large(self):
        result = keyword_summaries(self.large_summaries)
        expected = {
            "1": ["long", "test", "string", "contains", "different", "words", "including", "stop",
                  "words", "purpose", "test", "ensure", "keyword", "extraction", "works", "correctly", "large",
                  "inputs"],
            "2": ["long", "test", "string", "contains", "different", "words", "purpose", "test",
                  "robustness", "keyword", "extraction", "function"],
            "3": [ "example", "long", "test", "string", "keyword", "extraction", "function", "able", "handle",
                  "issues"],
            "4": ["short", "string"],
            "5": ["string", "numbers", "12345", "special", "characters"],
            "6": ["string", "mixed", "case", "words", "stop", "words", "punctuations"]
        }
        self.assertEqual(result, expected)

    def test_match_keywords(self):
        result = match_keywords(["keyword1", "keyword2"], ["keyword2", "keyword3"])
        expected = ["keyword2"]
        self.assertEqual(result, expected)

    def test_match_keywords_no_match(self):
        result = match_keywords(["keyword1"], ["keyword2", "keyword3"])
        expected = []
        self.assertEqual(result, expected)

    def test_match_keywords_empty(self):
        result = match_keywords([], [])
        expected = []
        self.assertEqual(result, expected)

    def test_create_kw_edges_valid(self):
        summary_kws = {"1": ["keyword1", "keyword2"], "2": ["keyword2", "keyword3"]}
        result = create_kw_edges(summary_kws)
        expected = [("1", "2", 0.5, ["keyword2"])]
        self.assertEqual(result, expected)

    def test_create_kw_edges_large(self):
        summary_kws = {
            "1": ["keyword1", "keyword2", "keyword3"],
            "2": ["keyword2", "keyword3", "keyword4"],
            "3": ["keyword3", "keyword4", "keyword5"],
            "4": ["keyword1", "keyword3", "keyword5"]
        }
        result = create_kw_edges(summary_kws)
        expected = [
            ("1", "2", 0.6666666666666666, ["keyword2", "keyword3"]),
            ("1", "3", 0.3333333333333333, ["keyword3"]),
            ("1", "4", 0.6666666666666666, ["keyword1", "keyword3"]),
            ("2", "3", 0.6666666666666666, ["keyword3", "keyword4"]),
            ("2", "4", 0.3333333333333333, ["keyword3"]),
            ("3", "4", 0.6666666666666666, ["keyword3", "keyword5"])
        ]
        self.assertEqual(result, expected)

    def test_create_kw_edges_empty(self):
        result = create_kw_edges({})
        expected = []
        self.assertEqual(result, expected)

    def test_build_kw_graph_valid(self):
        summary_kws = {"1": ["keyword1", "keyword2"], "2": ["keyword2", "keyword3"]}
        result = build_kw_graph(summary_kws)
        self.assertIsInstance(result, nx.MultiGraph)
        self.assertEqual(len(result.nodes), 2)
        self.assertEqual(len(result.edges), 1)

    def test_build_kw_graph_large(self):
        summary_kws = {
            "1": ["keyword1", "keyword2", "keyword3"],
            "2": ["keyword2", "keyword3", "keyword4"],
            "3": ["keyword3", "keyword4", "keyword5"],
            "4": ["keyword1", "keyword3", "keyword5"]
        }
        result = build_kw_graph(summary_kws)
        self.assertIsInstance(result, nx.MultiGraph)
        self.assertEqual(len(result.nodes), 4)
        self.assertEqual(len(result.edges), 6)

    def test_build_kw_graph_empty(self):
        result = build_kw_graph({})
        self.assertIsInstance(result, nx.MultiGraph)
        self.assertEqual(len(result.nodes), 0)
        self.assertEqual(len(result.edges), 0)

    def test_load_seg_similaritites_valid(self):
        with patch("builtins.open", mock_open(read_data=self.seg_similarity_csv)):
            result = load_seg_similaritites("fake_file.csv")
            expected = pd.read_csv(StringIO(self.seg_similarity_csv), index_col=0)
            pd.testing.assert_frame_equal(result, expected)

    def test_load_seg_similaritites_invalid(self):
        with self.assertRaises(pd.errors.ParserError):
            with patch("builtins.open", mock_open(read_data=self.invalid_seg_similarity_csv)):
                with patch("pandas.read_csv", side_effect=pd.errors.ParserError):
                    load_seg_similaritites("fake_file.csv")

    def test_create_seg_edges_valid(self):
        with patch("builtins.open", mock_open(read_data=self.seg_similarity_csv)):
            with patch("pandas.read_csv", return_value=pd.read_csv(StringIO(self.seg_similarity_csv), index_col=0)):
                result = create_seg_edges(self.valid_seg_ids, "fake_file.csv")
                expected = [
                    ("1", "2", 0.5, ("seg1", "seg2")),
                    ("1", "2", 0.2, ("seg1", "seg3")),
                    ("1", "2", 1.0, ("seg2", "seg2")),
                    ("1", "2", 0.3, ("seg2", "seg3"))
                ]
                self.assertEqual(result, expected)

    def test_create_seg_edges_large(self):
        with patch("builtins.open", mock_open(read_data=self.seg_similarity_csv)):
            with patch("pandas.read_csv", return_value=pd.read_csv(StringIO(self.seg_similarity_csv), index_col=0)):
                result = create_seg_edges(self.large_seg_ids, "fake_file.csv")
                expected = [
                    ('1', '2', 0.5, ('seg1', 'seg2')),
                     ('1', '2', 0.2, ('seg1', 'seg3')),
                     ('1', '2', 0.3, ('seg1', 'seg4')),
                     ('1', '2', 0.1, ('seg1', 'seg5')),
                     ('1', '2', 1.0, ('seg2', 'seg2')),
                     ('1', '2', 0.3, ('seg2', 'seg3')),
                     ('1', '2', 0.4, ('seg2', 'seg4')),
                     ('1', '2', 0.2, ('seg2', 'seg5')),
                     ('1', '2', 0.3, ('seg3', 'seg2')),
                     ('1', '2', 1.0, ('seg3', 'seg3')),
                     ('1', '2', 0.5, ('seg3', 'seg4')),
                     ('1', '2', 0.3, ('seg3', 'seg5')),
                     ('1', '2', 0.4, ('seg4', 'seg2')),
                     ('1', '2', 0.5, ('seg4', 'seg3')),
                     ('1', '2', 1.0, ('seg4', 'seg4')),
                     ('1', '2', 0.6, ('seg4', 'seg5')),
                     ('1', '3', 0.2, ('seg1', 'seg3')),
                     ('1', '3', 0.3, ('seg1', 'seg4')),
                     ('1', '3', 0.1, ('seg1', 'seg5')),
                     ('1', '3', 1.0, ('seg1', 'seg1')),
                     ('1', '3', 0.3, ('seg2', 'seg3')),
                     ('1', '3', 0.4, ('seg2', 'seg4')),
                     ('1', '3', 0.2, ('seg2', 'seg5')),
                     ('1', '3', 0.5, ('seg2', 'seg1')),
                     ('1', '3', 1.0, ('seg3', 'seg3')),
                     ('1', '3', 0.5, ('seg3', 'seg4')),
                     ('1', '3', 0.3, ('seg3', 'seg5')),
                     ('1', '3', 0.2, ('seg3', 'seg1')),
                     ('1', '3', 0.5, ('seg4', 'seg3')),
                     ('1', '3', 1.0, ('seg4', 'seg4')),
                     ('1', '3', 0.6, ('seg4', 'seg5')),
                     ('1', '3', 0.3, ('seg4', 'seg1')),
                     ('1', '4', 1.0, ('seg1', 'seg1')),
                     ('1', '4', 0.2, ('seg1', 'seg3')),
                     ('1', '4', 0.1, ('seg1', 'seg5')),
                     ('1', '4', 0.5, ('seg1', 'seg2')),
                     ('1', '4', 0.5, ('seg2', 'seg1')),
                     ('1', '4', 0.3, ('seg2', 'seg3')),
                     ('1', '4', 0.2, ('seg2', 'seg5')),
                     ('1', '4', 1.0, ('seg2', 'seg2')),
                     ('1', '4', 0.2, ('seg3', 'seg1')),
                     ('1', '4', 1.0, ('seg3', 'seg3')),
                     ('1', '4', 0.3, ('seg3', 'seg5')),
                     ('1', '4', 0.3, ('seg3', 'seg2')),
                     ('1', '4', 0.3, ('seg4', 'seg1')),
                     ('1', '4', 0.5, ('seg4', 'seg3')),
                     ('1', '4', 0.6, ('seg4', 'seg5')),
                     ('1', '4', 0.4, ('seg4', 'seg2')),
                     ('2', '3', 0.3, ('seg2', 'seg3')),
                     ('2', '3', 0.4, ('seg2', 'seg4')),
                     ('2', '3', 0.2, ('seg2', 'seg5')),
                     ('2', '3', 0.5, ('seg2', 'seg1')),
                     ('2', '3', 1.0, ('seg3', 'seg3')),
                     ('2', '3', 0.5, ('seg3', 'seg4')),
                     ('2', '3', 0.3, ('seg3', 'seg5')),
                     ('2', '3', 0.2, ('seg3', 'seg1')),
                     ('2', '3', 0.5, ('seg4', 'seg3')),
                     ('2', '3', 1.0, ('seg4', 'seg4')),
                     ('2', '3', 0.6, ('seg4', 'seg5')),
                     ('2', '3', 0.3, ('seg4', 'seg1')),
                     ('2', '3', 0.3, ('seg5', 'seg3')),
                     ('2', '3', 0.6, ('seg5', 'seg4')),
                     ('2', '3', 1.0, ('seg5', 'seg5')),
                     ('2', '3', 0.1, ('seg5', 'seg1')),
                     ('2', '4', 0.5, ('seg2', 'seg1')),
                     ('2', '4', 0.3, ('seg2', 'seg3')),
                     ('2', '4', 0.2, ('seg2', 'seg5')),
                     ('2', '4', 1.0, ('seg2', 'seg2')),
                     ('2', '4', 0.2, ('seg3', 'seg1')),
                     ('2', '4', 1.0, ('seg3', 'seg3')),
                     ('2', '4', 0.3, ('seg3', 'seg5')),
                     ('2', '4', 0.3, ('seg3', 'seg2')),
                     ('2', '4', 0.3, ('seg4', 'seg1')),
                     ('2', '4', 0.5, ('seg4', 'seg3')),
                     ('2', '4', 0.6, ('seg4', 'seg5')),
                     ('2', '4', 0.4, ('seg4', 'seg2')),
                     ('2', '4', 0.1, ('seg5', 'seg1')),
                     ('2', '4', 0.3, ('seg5', 'seg3')),
                     ('2', '4', 1.0, ('seg5', 'seg5')),
                     ('2', '4', 0.2, ('seg5', 'seg2')),
                     ('3', '4', 0.2, ('seg3', 'seg1')),
                     ('3', '4', 1.0, ('seg3', 'seg3')),
                     ('3', '4', 0.3, ('seg3', 'seg5')),
                     ('3', '4', 0.3, ('seg3', 'seg2')),
                     ('3', '4', 0.3, ('seg4', 'seg1')),
                     ('3', '4', 0.5, ('seg4', 'seg3')),
                     ('3', '4', 0.6, ('seg4', 'seg5')),
                     ('3', '4', 0.4, ('seg4', 'seg2')),
                     ('3', '4', 0.1, ('seg5', 'seg1')),
                     ('3', '4', 0.3, ('seg5', 'seg3')),
                     ('3', '4', 1.0, ('seg5', 'seg5')),
                     ('3', '4', 0.2, ('seg5', 'seg2')),
                     ('3', '4', 1.0, ('seg1', 'seg1')),
                     ('3', '4', 0.2, ('seg1', 'seg3')),
                     ('3', '4', 0.1, ('seg1', 'seg5')),
                     ('3', '4', 0.5, ('seg1', 'seg2'))
                ]
                self.assertEqual(result, expected)

    def test_create_seg_edges_empty(self):
        with patch("builtins.open", mock_open(read_data=self.seg_similarity_csv)):
            with patch("pandas.read_csv", return_value=pd.read_csv(StringIO(self.seg_similarity_csv), index_col=0)):
                result = create_seg_edges(self.empty_seg_ids, "fake_file.csv")
                expected = []
                self.assertEqual(result, expected)

    def test_build_seg_graph_valid(self):
        with patch("builtins.open", mock_open(read_data=self.seg_similarity_csv)):
            with patch("pandas.read_csv", return_value=pd.read_csv(StringIO(self.seg_similarity_csv), index_col=0)):
                result = build_seg_graph(self.valid_seg_ids, "fake_file.csv")
                self.assertIsInstance(result, nx.MultiGraph)
                self.assertEqual(len(result.nodes), 2)
                self.assertEqual(len(result.edges), 4)

    def test_build_seg_graph_large(self):
        with patch("builtins.open", mock_open(read_data=self.seg_similarity_csv)):
            with patch("pandas.read_csv", return_value=pd.read_csv(StringIO(self.seg_similarity_csv), index_col=0)):
                result = build_seg_graph(self.large_seg_ids, "fake_file.csv")
                self.assertIsInstance(result, nx.MultiGraph)
                self.assertEqual(len(result.nodes), 4)
                self.assertEqual(len(result.edges), 96)

    def test_build_seg_graph_empty(self):
        with patch("builtins.open", mock_open(read_data=self.seg_similarity_csv)):
            with patch("pandas.read_csv", return_value=pd.read_csv(StringIO(self.seg_similarity_csv), index_col=0)):
                result = build_seg_graph(self.empty_seg_ids, "fake_file.csv")
                self.assertIsInstance(result, nx.MultiGraph)
                self.assertEqual(len(result.nodes), 0)
                self.assertEqual(len(result.edges), 0)

    def test_update_probabilities_valid(self):
        with patch("builtins.open", mock_open(read_data=self.seg_similarity_csv)):
            with patch("pandas.read_csv", return_value=pd.read_csv(StringIO(self.seg_similarity_csv), index_col=0)):
                g_model, g_desc = update_probabilities(self.valid_seg_ids, "fake_file.csv", self.desc_kws)
                self.assertIsInstance(g_model, nx.MultiGraph)
                self.assertIsInstance(g_desc, nx.MultiGraph)
                self.assertGreater(len(g_model.nodes), 0)
                self.assertGreater(len(g_desc.nodes), 0)

    def test_update_probabilities_large(self):
        with patch("builtins.open", mock_open(read_data=self.seg_similarity_csv)):
            with patch("pandas.read_csv", return_value=pd.read_csv(StringIO(self.seg_similarity_csv), index_col=0)):
                g_model, g_desc = update_probabilities(self.large_seg_ids, "fake_file.csv", self.large_desc_kws)
                self.assertIsInstance(g_model, nx.MultiGraph)
                self.assertIsInstance(g_desc, nx.MultiGraph)
                self.assertEqual(len(g_model.nodes), 4)
                self.assertEqual(len(g_desc.nodes), 4)

    def test_update_probabilities_empty(self):
        with patch("builtins.open", mock_open(read_data=self.seg_similarity_csv)):
            with patch("pandas.read_csv", return_value=pd.read_csv(StringIO(self.seg_similarity_csv), index_col=0)):
                g_model, g_desc = update_probabilities(self.empty_seg_ids, "fake_file.csv", self.empty_summaries)
                self.assertIsInstance(g_model, nx.MultiGraph)
                self.assertIsInstance(g_desc, nx.MultiGraph)
                self.assertEqual(len(g_model.nodes), 0)
                self.assertEqual(len(g_desc.nodes), 0)


if __name__ == "__main__":
    unittest.main()
