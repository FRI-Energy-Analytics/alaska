import unittest
from .keyword_tree import Alias, search, make_tree, search_child
from .predict_from_model import make_prediction


class TestTree(unittest.TestCase):
    """
    Unit tests for well log mnemonics parser
    """
    def test_make_tree(self):
        """
        Test that it can build the keyword tree
        """
        root = make_tree()
        self.assertEqual(root.child[0].key, "caliper")

    def test_search(self):
        """
        Test that it can search the keyword tree
        """
        result = search(make_tree(), "gr")
        self.assertEqual(result, "gamma ray")

    def test_search_child(self):
        """
        Test that it can search nodes of the keyword tree
        """
        result = search_child(make_tree(), "density porosity")
        self.assertEqual(result, "density porosity")

    def test_parse(self):  # 1000080059
        """
        Test that Aliaser can parse las file
        """
        aliaser = Alias()
        result = aliaser.parse("data/testcase1.las")
        self.assertEqual(result, ({"dept": "depth", "gr": "gamma ray"}, []))

    def test_dictionary_parse(self):
        """
        Test that dictionary parser in Aliaser parses and returns correct labels
        """
        aliaser = Alias(keyword_extractor=False, model=False)
        result = aliaser.parse("data/testcase1.las")
        self.assertEqual(result, ({"dept": "depth", "gr": "gamma ray"}, []))

    def test_keyword_parse(self):  # 3725733C.las
        """
        Test that keyword parser in Aliaser parses and returns correct labels
        """
        aliaser = Alias(dictionary=False, model=False)
        result = aliaser.parse("data/testcase2.las")
        self.assertEqual(result, ({"dphi": "density porosity", "cali": "caliper"}, []))

    def test_model_parse(self):
        """
        Test that model in Aliaser parses and returns correct predictions
        """
        aliaser = Alias(dictionary=False, keyword_extractor=False)
        result = aliaser.parse("data/testcase3.las")
        self.assertEqual(
            result, ({"qn": "near quality", "dphi": "density porosity"}, [])
        )

    def test_make_prediction(self):
        """
        Test that make prediction works
        """
        result = make_prediction("data/testcase4.gz")
        self.assertEqual(result, ({"qn": "near quality"}))


if __name__ == "__main__":
    unittest.main()
