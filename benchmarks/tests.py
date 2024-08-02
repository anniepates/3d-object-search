import unittest
import pandas as pd
import cProfile
import pstats
import io
from unittest.mock import patch
from memory_profiler import profile
import build_graph


# Decorator for profiling
def profile_decorator(func):
    def wrapper(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        result = func(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
        return result

    return wrapper


class TestKeywordExtraction(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Setup mock data for testing
        cls.mock_data = {
            'id': ['1', '2', '3'],
            'content': [
                "This is a 3D printable prosthetic finger.",
                "A pillbox with eight compartments.",
                "A round pillbox for one month supply."
            ]
        }
        cls.mock_df = pd.DataFrame(cls.mock_data)
        cls.seg_similarity_data = {
            'segment1': [1, 0.5, 0.2],
            'segment2': [0.5, 1, 0.3],
            'segment3': [0.2, 0.3, 1]
        }
        cls.seg_similarity_df = pd.DataFrame(cls.seg_similarity_data, index=['segment1', 'segment2', 'segment3'])

    @patch('pandas.read_csv')
    @profile
    @profile_decorator
    def test_create_dictionary(self, mock_read_csv):
        mock_read_csv.return_value = self.mock_df
        result = build_graph.create_dictionary('dummy_file.csv')
        expected = {'1': "This is a 3D printable prosthetic finger.", '2': "A pillbox with eight compartments.",
                    '3': "A round pillbox for one month supply."}
        self.assertEqual(result, expected)

    @profile
    @profile_decorator
    def test_keyword_helper(self):
        content = "This is a 3D printable prosthetic finger."
        result = build_graph.keyword_helper(content)
        expected = ['3d', 'printable', 'prosthetic', 'finger']
        self.assertEqual(result, expected)

    @profile
    @profile_decorator
    def test_keyword_summaries(self):
        summaries = {'1': "This is a 3D printable prosthetic finger.", '2': "A pillbox with eight compartments.",
                     '3': "A round pillbox for one month supply."}
        result = build_graph.keyword_summaries(summaries)
        expected = {
            '1': ['3d', 'printable', 'prosthetic', 'finger'],
            '2': ['pillbox', 'eight', 'compartments'],
            '3': ['round', 'pillbox', 'month', 'supply']
        }
        self.assertEqual(result, expected)

    @profile
    @profile_decorator
    def test_build_kw_graph(self):
        kws = {
            '1': ['3d', 'printable', 'prosthetic', 'finger'],
            '2': ['pillbox', 'eight', 'compartments'],
            '3': ['round', 'pillbox', 'month', 'supply']
        }
        G = build_graph.build_kw_graph(kws)
        self.assertEqual(len(G.nodes), 3)
        self.assertGreater(len(G.edges), 0)

    @patch('pandas.read_csv')
    @profile
    @profile_decorator
    def test_update_probabilities(self, mock_read_csv):
        mock_read_csv.return_value = self.seg_similarity_df
        seg_ids = {
            'model1': ['segment1', 'segment2'],
            'model2': ['segment2', 'segment3']
        }
        desc_kws = {
            'model1': ['3d', 'printable'],
            'model2': ['pillbox', 'round']
        }
        g_model, g_desc = build_graph.update_probabilities(seg_ids, 'dummy_file.csv', desc_kws)
        self.assertEqual(len(g_model.nodes), 2)
        self.assertEqual(len(g_desc.nodes), 2)
        self.assertGreater(len(g_model.edges), 0)
        self.assertGreater(len(g_desc.edges), 0)

    @profile
    @profile_decorator
    def test_search_returns_correct_objects(self):
        summaries = {'1': "This is a 3D printable prosthetic finger.", '2': "A pillbox with eight compartments.",
                     '3': "A round pillbox for one month supply."}
        kws = build_graph.keyword_summaries(summaries)
        G = build_graph.build_kw_graph(kws)
        search_keyword = 'prosthetic'
        results = [node for node, attrs in G.nodes.data() if search_keyword in attrs['kws']]
        self.assertIn('1', results)
        self.assertNotIn('2', results)
        self.assertNotIn('3', results)

    @profile
    @profile_decorator
    def test_keyword_suggestions(self):
        descriptions = ["This 3D printable prosthetic finger is ideal.", "A pillbox with eight compartments."]
        expected_keywords = ['3d', 'printable', 'prosthetic', 'finger', 'pillbox', 'eight', 'compartments']
        all_keywords = []
        for desc in descriptions:
            all_keywords.extend(build_graph.keyword_helper(desc))
        self.assertTrue(set(expected_keywords).issubset(set(all_keywords)))

    @profile
    @profile_decorator
    def test_component_search(self):
        seg_ids = {
            'model1': ['handle', 'cup'],
            'model2': ['wheel', 'frame']
        }
        desc_kws = {
            'model1': ['mug', 'handle'],
            'model2': ['bicycle', 'wheel']
        }
        G = build_graph.build_kw_graph(desc_kws)
        search_keyword = 'mug'
        component_results = [node for node, attrs in G.nodes.data() if search_keyword in attrs['kws']]
        self.assertIn('model1', component_results)
        self.assertNotIn('model2', component_results)

    @profile
    @profile_decorator
    def test_edge_case_empty_description(self):
        descriptions = ["", "A pillbox with eight compartments."]
        expected_keywords = ['pillbox', 'eight', 'compartments']
        all_keywords = []
        for desc in descriptions:
            all_keywords.extend(build_graph.keyword_helper(desc))
        self.assertTrue(set(expected_keywords).issubset(set(all_keywords)))

    @profile
    @profile_decorator
    def test_edge_case_large_input(self):
        large_description = " ".join(["word"] * 10000)
        result = build_graph.keyword_helper(large_description)
        expected = ['word']
        self.assertEqual(result, expected)


if __name__ == '__main__':
    unittest.main()
