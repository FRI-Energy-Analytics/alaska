# Copyright (c) 2021 The AlasKA Developers.
# Distributed under the terms of the MIT License.
# SPDX-License_Identifier: MIT

# MIT License
# Copyright (c) 2018 Yimai Fang

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Code modified from Yimai Fang's seq2seq-summarizer
# repository: https://github.com/ymfa/seq2seq-summarizer

# crains.json file modified from Welly
# Copyright (c) 2021 Agile Scientific
# Distributed under the Apache 2.0 License (see bottom of file)
# SPDX-License_Identifier: Apache-2.0
"""
Tests for Alaska's parser classes and functions
"""
from pathlib import Path
import matplotlib.pyplot as plt
import pytest
from ..keyword_tree import Alias, search, make_tree, search_child, Node
from ..predict_from_model import make_prediction
from ..get_data_path import get_data_path


test_case_1 = Path("alaska/data/testcase1.las")
test_case_2 = Path("alaska/data/testcase2.las")
test_case_3 = Path("alaska/data/testcase3.las")
test_case_4 = str(Path("alaska/data/testcase4.gz").resolve())
test_case_5 = Path("alaska/data/testcase5.las")
test_case_6 = Path("alaska/data/722319B.las")
test_custom_dict_csv = str(get_data_path("custom_search.csv"))
test_custom_dict_json = str(get_data_path("crains.json"))
test_dir_1 = Path("alaska/data/")


def test_make_tree():
    """
    Test that it can build the keyword tree
    """
    root = make_tree()
    assert root.child[0].key == "caliper"


def test_search():
    """
    Test that it can search the keyword tree
    """
    result = search(make_tree(), "gr")
    assert result == "gamma ray"


def test_search_child():
    """
    Test that it can search nodes of the keyword tree
    """
    result = search_child(make_tree(), "density porosity")
    assert result == "density porosity"


def test_search_child_2():
    """
    Test that search of empty nodes returns None
    """
    empty_tree = Node(None)
    result = search_child(empty_tree, "density porosity")
    assert result is None


def test_search_child_3():
    """
    Test that search nodes for an non-existent description returns None
    """
    result = search_child(make_tree(), "not found")
    assert result is None


def test_parse():  # 1000080059
    """
    Test that Aliaser can parse las file
    """
    aliaser = Alias()
    result = aliaser.parse(test_case_1)
    assert result == ({"depth": ["DEPT"], "gamma ray": ["GR"]}, [])


def test_parse_2():
    """
    Test that Aliaser can parse las file with an empty mnemonic
    """
    aliaser = Alias()
    result = aliaser.parse(test_case_5)
    assert result == ({"depth": ["DEPT"], "gamma ray": ["GR"]}, ["empty"])


def test_parse_custom_csv():
    """
    Test that the Aliaser can load and parse with a custom csv dictionary
    """
    aliaser = Alias(
        dictionary=True, custom_dict=test_custom_dict_csv, keyword_extractor=False
    )
    result, not_found = aliaser.parse(test_case_6)
    assert result == {
        "caliper": ["CALI"],
        "density porosity": ["DPHI"],
        "density correction": ["DRHO"],
        "gamma ray": ["GR"],
        "deep conductivity": ["HDCN"],
        "deep resistivity": ["HDRS"],
        "medium resistivity": ["HMRS"],
    }
    assert not_found[0] == "dept"
    assert not_found[-1] == "tens"


def test_parse_custom_json():
    """
    Test that the Aliaser can load and parse with a custom json dictionary
    """
    aliaser = Alias(
        dictionary=True, custom_dict=test_custom_dict_json, keyword_extractor=False
    )
    result, not_found = aliaser.parse(test_case_6)
    assert result == {
        "cal": ["CALI"],
        "phid": ["DPHI"],
        "dcor": ["DRHO"],
        "gr": ["GR"],
        "phin": ["NPHI"],
        "pe": ["PE"],
        "dens": ["RHOB"],
        "sp": ["SP"],
    }
    assert not_found[0] == "dept"
    assert not_found[-1] == "tens"


def test_custom_fail():
    """
    Test that the Aliaser prints error message for wrong custom dict data type
    """
    aliaser = Alias(
        dictionary=True, custom_dict=str(test_case_6), keyword_extractor=False
    )
    with pytest.raises(IOError):
        aliaser.parse(test_case_6)


def test_parse_directory_1():
    """
    Test that Aliaser can parse a directory of las files
    """
    aliaser = Alias()
    aliased, not_aliased = aliaser.parse_directory(test_dir_1)
    have1 = aliased.get("density porosity", None)
    have2 = aliased.get("depth", None)
    have3 = aliased.get("gamma ray", None)

    # Aliased
    assert have1 == ["DPHI"]
    assert have2 == ["DEPT"]
    assert have3 == ["GR"]

    # Not Aliased
    assert "qn" in not_aliased
    assert "empty" in not_aliased


def test_parse_directory_2():
    """
    Test that Aliaser can parse a directory of las files and use the model
    parser
    """
    aliaser = Alias(dictionary=False, keyword_extractor=False, model=True)
    aliased, not_aliased = aliaser.parse_directory(test_dir_1)
    have1 = aliased.get("density porosity", None)
    have2 = aliased.get("medium conductivity", None)
    assert have1 == ["DPHI"]
    assert have2 == ["DEPT"]
    assert "empty" in not_aliased


def test_parse_directory_3():
    """
    Test that Aliaser can parse .LAS and .las
    """
    aliaser = Alias()
    aliased, _ = aliaser.parse_directory(test_dir_1)
    assert len(aliased.keys()) > 0


def test_dictionary_parse_1():
    """
    Test that dictionary parser in Aliaser parses and returns correct labels
    """
    aliaser = Alias(keyword_extractor=False, model=False)
    result = aliaser.parse(test_case_1)
    assert result == ({"depth": ["DEPT"], "gamma ray": ["GR"]}, [])


def test_dictionary_parse_3():
    """
    Test that dictionary parser in Aliaser parses and returns correct labels
    with one item aliased and one item not aliased.
    """
    aliaser = Alias(keyword_extractor=False, model=False)
    aliased, not_aliased = aliaser.parse(test_case_3)

    assert len(aliased) == 1
    assert "density porosity" in aliased
    assert len(not_aliased) == 1
    assert "qn" in not_aliased


def test_keyword_parse_1():
    """
    Test that keyword parser in Aliaser parses and returns correct labels
    """
    aliaser = Alias(dictionary=False, model=False)
    aliased, not_aliased = aliaser.parse(test_case_1)

    assert len(aliased) == 1
    assert "gamma ray" in aliased
    assert len(not_aliased) == 1
    assert "dept" in not_aliased


def test_keyword_parse_2():  # 3725733C.las
    """
    Test that keyword parser in Aliaser parses and returns correct labels
    """
    aliaser = Alias(dictionary=False, model=False)
    result = aliaser.parse(test_case_2)
    assert result == ({"density porosity": ["DPHI"], "caliper": ["CALI"]}, [])


def test_keyword_parse_3():
    """
    Test that keyword parser in Aliaser successfully removes found items from
    the not_found list
    """
    aliaser = Alias(dictionary=False, model=False)
    aliaser.not_found.extend(["gr", "no such thing"])
    found, not_found = aliaser.parse(test_case_5)

    have1 = found.get("gamma ray", None)

    assert have1 == ["GR"]
    assert "gr" not in not_found
    assert "gr" not in aliaser.not_found
    assert "no such thing" in not_found
    assert "no such thing" in aliaser.not_found


def test_dictionary_and_keyword_parse_1():
    """
    Test that keyword parser in Aliaser parses and returns correct labels
    """
    aliaser = Alias(model=False)
    aliased, not_aliased = aliaser.parse(test_case_1)

    assert aliased == {"depth": ["DEPT"], "gamma ray": ["GR"]}
    assert len(not_aliased) == 0


def test_dictionary_and_keyword_parse_3():
    """
    Test that keyword parser in Aliaser parses and returns correct labels
    """
    aliaser = Alias(model=False)
    aliased, not_aliased = aliaser.parse(test_case_3)

    assert len(aliased) == 1
    assert "density porosity" in aliased
    assert len(not_aliased) == 1
    assert "qn" in not_aliased


def test_model_parse():
    """
    Test that model in Aliaser parses and returns correct predictions
    """
    aliaser = Alias(dictionary=False, keyword_extractor=False, model=True)
    result = aliaser.parse(test_case_3)
    assert result == ({"near quality": ["QN"], "density porosity": ["DPHI"]}, [])


def test_model_parse_2():
    """
    Test that model in Aliaser parses and returns correct predictions and
    removes found items from the not_found list

    expected data
    aliased == {"near quality": ["QN"], "density porosity": ["DPHI"]}
    not_aliased == []
    """
    aliaser = Alias(dictionary=True, keyword_extractor=False, model=True)
    aliased, not_aliased = aliaser.parse(test_case_3)

    have1 = aliased.get("density porosity", None)
    have2 = aliased.get("near quality", None)

    assert have1 == ["DPHI"]
    assert have2 == ["QN"]
    assert "qn" not in not_aliased
    assert "qn" not in aliaser.not_found


def test_make_prediction():
    """
    Test that make prediction works
    """
    result = make_prediction(test_case_4)
    assert result[0] == {"qn": "near quality"}
    assert result[1]["qn"] == pytest.approx(0.8421125945781427, rel=1e-4)


def test_heatmap():
    """
    Test that the aliaser can create a heatmap
    """
    aliaser = Alias()
    aliaser.parse(test_case_6)
    aliaser.heatmap()
    plt.gcf().canvas.draw()


"""
Welly license for crains.json file
                           Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

     "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "{}"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright {yyyy} {name of copyright owner}

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""
