import os.path
import gzip
import pandas as pd
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import lasio
from predict_from_model import make_prediction


class Node:
    def __init__(self, key):
        self.key = key
        self.child = []


def make_tree():
    """
    :param: None
    :return: m-ary tree of keywords that forms keyword extractor tree
    Generates keyword extractor tree
    """
    root = Node(None)
    df = (
        pd.read_csv("data/original_lowered.csv")
        .drop("Unnamed: 0", 1)
        .reset_index(drop=True)
    )
    arr = df.label.unique()
    cali_arr = ["calibration", "diameter", "radius"]
    time_arr = ["time", "delta-t", "dt", "delta"]
    gr_arr = ["gamma", "ray", "gr", "gamma-ray"]
    sp_arr = ["sp", "spontaneous", "potential"]
    d_arr = ["correction", "porosity"]
    p_arr = ["density", "neutron", "sonic"]
    p2_arr = ["dolomite", "limestone"]
    r_arr = ["deep", "shallow", "medium"]
    sr_arr = ["a10", "a20", "ae10", "ae20", "10in", "20in"]
    mr_arr = ["a30", "ae30", "30in"]
    dr_arr = ["a60", "a90", "ae60", "ae90", "60in", "90in"]
    j = 0
    for i in arr:
        root.child.append(Node(i))
        root.child[j].child.append(Node(i))
        j += 1
    for i in cali_arr:
        root.child.append(Node(i))
        root.child[j].child.append(Node("caliper"))
        j += 1
    for i in time_arr:
        root.child.append(Node(i))
        root.child[j].child.append(Node("sonic travel time"))
        j += 1
    for i in gr_arr:
        root.child.append(Node(i))
        root.child[j].child.append(Node("gamma ray"))
        j += 1
    for i in sp_arr:
        root.child.append(Node(i))
        root.child[j].child.append(Node("spontaneous potential"))
        j += 1
    root.child.append(Node("photoelectric"))
    root.child[j].child.append(Node("photoelectric effect"))
    j += 1
    root.child.append(Node("bit"))
    root.child[j].child.append(Node("bit size"))
    j += 1
    for i in sr_arr:
        root.child.append(Node(i))
        root.child[j].child.append(Node("shallow resistivity"))
        j += 1
    for i in mr_arr:
        root.child.append(Node(i))
        root.child[j].child.append(Node("medium resistivity"))
        j += 1
    for i in dr_arr:
        root.child.append(Node(i))
        root.child[j].child.append(Node("deep resistivity"))
        j += 1
    root.child.append(Node("density"))
    k = 0
    for i in d_arr:
        root.child[j].child.append(Node(i))
        st = "density " + str(i)
        root.child[j].child[k].child.append(Node(st))
        k += 1
    root.child[j].child.append(Node("bulk"))
    root.child[j].child[k].child.append(Node("bulk density"))
    root.child.append(Node("porosity"))
    j += 1
    k = 0
    for i in p_arr:
        root.child[j].child.append(Node(i))
        st = str(i) + " porosity"
        root.child[j].child[k].child.append(Node(st))
        k += 1
    for i in p2_arr:
        root.child[j].child.append(Node(i))
        st = "density porosity"
        root.child[j].child[k].child.append(Node(st))
        k += 1
    root.child.append(Node("conductivity"))
    j += 1
    k = 0
    for i in r_arr:
        root.child[j].child.append(Node(i))
        st = str(i) + " conductivity"
        root.child[j].child[k].child.append(Node(st))
        k += 1
    root.child.append(Node("resistivity"))
    j += 1
    k = 0
    for i in r_arr:
        root.child[j].child.append(Node(i))
        st = str(i) + " resistivity"
        root.child[j].child[k].child.append(Node(st))
        k += 1
    for i in sr_arr:
        root.child[j].child.append(Node(i))
        st = "shallow resistivity"
        root.child[j].child[k].child.append(Node(st))
        k += 1
    for i in mr_arr:
        root.child[j].child.append(Node(i))
        st = "medium resistivity"
        root.child[j].child[k].child.append(Node(st))
        k += 1
    for i in dr_arr:
        root.child[j].child.append(Node(i))
        st = "deep resistivity"
        root.child[j].child[k].child.append(Node(st))
        k += 1
    root.child[j].child.append(Node("micro"))
    st = "micro resistivity"
    root.child[j].child[k].child.append(Node(st))
    root.child.append(Node("res"))
    j += 1
    k = 0
    for i in r_arr:
        root.child[j].child.append(Node(i))
        st = str(i) + " resistivity"
        root.child[j].child[k].child.append(Node(st))
        k += 1
    root.child.append(Node("cond"))
    j += 1
    k = 0
    for i in r_arr:
        root.child[j].child.append(Node(i))
        st = str(i) + " conductivity"
        root.child[j].child[k].child.append(Node(st))
        k += 1
    return root

def search(tree, description):
    """
    :param tree: m-ary keyword extractor tree
    :param description: mnemonic description from LAS file
    :return: none if keyword does not exist in tree, label if keyword exists in tree
    Search keyword extractor tree
    """
    arr = [tree]
    arr = [c for node in arr for c in node.child if c]
    for i in description.split():
        for node in arr:
            if i == node.key:
                return search_child(node, description)
    return None


def search_child(node, description):
    """
    :param node: node of m-ary keyword extractor tree
    :param description: mnemonic description from LAS file
    :return: none if keyword does not exist in tree, label if keyword exists in tree
    Search keyword extractor node
    """
    if len(node.child) < 1:
        return None
    elif len(node.child) == 1:
        return node.child[0].key
    else:
        for i in description.split():
            for child in node.child:
                if i == child.key:
                    return search_child(child, description)
    return None


class Alias:
    """
    :param dictionary: whether to use dictionary to make predictions on the labels
    :param keyword_extractor: whether to use keyword extractor to make predictions on the labels
    :param model: whether to use model to make predictions on the labels
    :return: one dictionary containing mnemonics and labels, one list containing mnemonics that can't be aliased
    Parses LAS file and returns parsed mnemonics with labels
    """

    # Constructor
    def __init__(self, dictionary=True, keyword_extractor=True, model=True, prob_cutoff=.5):
        self.dictionary = dictionary
        self.keyword_extractor = keyword_extractor
        self.prob_cutoff = prob_cutoff
        self.model = model
        self.duplicate, self.not_found = [], []
        self.method, self.probability, self.mnem = [], [], []
        self.output = {}

    def parse(self, path):
        """
        :param path: path to LAS file to be aliased
        :return: one dictionary containing mnemonics and labels, one list containing mnemonics that can't be aliased
        Parses LAS file and call parsers accordingly
        """
        las = lasio.read(path)
        mnem, desc = [], []
        for key in las.keys():
            mnem.append(key.lower())
            if str(las.curves[key].descr) == "" and str(las.curves[key].value) == "":
                desc.append("None")
            else:
                desc.append(str(las.curves[key].descr).lower())
        print("Reading {} mnemonics...".format(len(mnem)))
        if self.dictionary is True:
            self.dictionary_parse(mnem)
        if self.keyword_extractor is True:
            self.keyword_parse(mnem, desc)
        if self.model is True:
            df = self.make_df(path)
            self.model_parse(df)
        formatted_output = {}
        for key, val in self.output.items():  
            formatted_output.setdefault(val, []).append(key.upper()) 
        return formatted_output, self.not_found

    def parse_directory(self, directory):
        """
        :param path: path to directory containing LAS files
        :return: one dictionary containing mnemonics and labels, one list containing mnemonics that can't be aliased
        Parses LAS files and call parsers accordingly
        """
        comprehensive_dict = {}
        comprehensive_not_found = []
        for filename in os.listdir(directory):
            if filename.endswith(".las"):
                path = os.path.join(directory, filename)
                las = lasio.read(path)
                mnem, desc = [], []
                for key in las.keys():
                    mnem.append(key.lower())
                    if str(las.curves[key].descr) == "" and str(las.curves[key].value) == "":
                        desc.append("None")
                    else:
                        desc.append(str(las.curves[key].descr).lower())
                print("Reading {} mnemonics from {}...".format(len(mnem),filename))
                if self.dictionary is True:
                    self.dictionary_parse(mnem)
                if self.keyword_extractor is True:
                    self.keyword_parse(mnem, desc)
                if self.model is True:
                    df = self.make_df(path)
                    self.model_parse(df)
                comprehensive_dict.update(self.output)
                comprehensive_not_found.extend(self.not_found)
                self.output = {}
                self.duplicate, self.not_found = [], []
        formatted_output = {}
        for key, val in comprehensive_dict.items():  
            formatted_output.setdefault(val, []).append(key.upper()) 
        return formatted_output, comprehensive_not_found

    def heatmap(self):
        df = pd.DataFrame(
            {'method': self.method,
            'mnem': self.mnem,
            'prob': self.probability
            })
        result = df.pivot(index='method',columns='mnem',values='prob')
        fig = sns.heatmap(result)
        return fig

    def dictionary_parse(self, mnem):
        """
        :param mnem: list of mnemonics
        :return: None
        Find exact matches of mnemonics in mnemonic dictionary
        """
        df = (
            pd.read_csv("data/comprehensive_dictionary.csv")
            .drop("Unnamed: 0", 1)
            .reset_index(drop=True)
        )
        print("Alasing with dictionary...")
        dic = df.apply(lambda x: x.astype(str).str.lower())
        index = 0
        for i in mnem:
            if i in dic.mnemonics.unique():
                key = dic.loc[dic["mnemonics"] == i, "label"].iloc[0]  # can be reduced?
                self.output[i] = key
                self.duplicate.append(index)
                self.mnem.append(i)
                self.probability.append(1)
                self.method.append("dictionary")
            index += 1
        print("Aliased {} mnemonics with dictionary".format(index-1))

    def keyword_parse(self, mnem, desc):
        """
        :param mnem: list of mnemonics
        :param desc: list of descriptions
        :return: None
        Find exact labels of mnemonics with descriptions that can be filtered through keyword extractor tree
        """
        Tree = make_tree()
        new_desc = [v for i, v in enumerate(desc) if i not in self.duplicate]
        new_mnem = [v for i, v in enumerate(mnem) if i not in self.duplicate]
        index = 0
        print("Alasing with keyword extractor...")
        for i in new_desc:
            key = search(Tree, i)
            if key == None:
                self.not_found.append(new_mnem[index])
            else:
                self.output[new_mnem[index]] = key
                self.mnem.append(new_mnem[index])
                self.probability.append(1)
                self.method.append("keyword")
            index += 1
        print("Aliased {} mnemonics with keyword extractor".format(index-1))

    def model_parse(self, df):
        """
        :param df: dataframe of curves
        :return: None
        Make predictions using pointer generator
        """
        print("Alasing with pointer generator...")
        path = self.build_test(df)
        new_dictionary, predicted_prob = make_prediction(path)
        for key, value in predicted_prob.items():
            if float(value) >= self.prob_cutoff:
                self.output[key]=new_dictionary[key]
                self.mnem.append(key)
                self.probability.append(float(value))
                self.method.append("model")
            else:
                self.not_found.append(key)
        print("Aliased {} mnemonics with pointer generator".format(len(predicted_prob)))

    def build_test(self, df):
        """
        :param df: dataframe of curves
        :return: compressed file of summaries used to generate labels
        Build input file for pointer generator
        """
        data_path = "data/"
        test_out = gzip.open(os.path.join(data_path, "input.gz"), "wt")
        for i in range(len(df)):
            fout = test_out
            lst = [df.description[i], df.units[i], df.mnemonics[i]]
            summary = [df.mnemonics[i]]
            fout.write(" ".join(lst) + "\t" + " ".join(summary) + "\n")
            fout.flush()
        test_out.close()
        return os.path.join(data_path, "input.gz")

    def make_df(self, path):
        """
        :param path: path to the LAS file
        :return: dataframe of curves
        Build dataframe for pointer generator
        """
        mnem, description, unit = [], [], []
        las = lasio.read(path)
        if self.dictionary is not True and self.keyword_extractor is not True:
            for key in las.keys():
                mnem.append(key.lower())
                description.append(str(las.curves[key].descr).lower())
                unit.append(str(las.curves[key].unit).lower())
        else:
            for i in self.not_found:
                mnem.append(i)
                description.append(str(las.curves[i].descr).lower())
                unit.append(str(las.curves[i].unit).lower())
        output_df = pd.DataFrame(
            {"mnemonics": mnem, "description": description, "units": unit}
        )
        return output_df
