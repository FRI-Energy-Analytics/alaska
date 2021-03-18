# Copyright (c) 2021 The AlasKA Developers.
# Distributed under the terms of the MIT License.
# SPDX-License_Identifier: MIT
"""
Tests for alaska/_version.py
"""
import pytest
from .._version import versions_from_parentdir, NotThisMethod


def test_versions_from_parentdir():
    """
    Basic test of empty parameters
    """
    res = versions_from_parentdir("", "", "")
    assert isinstance(res, dict)
    assert res == {
        "version": "",
        "full-revisionid": None,
        "dirty": False,
        "error": None,
        "date": None,
    }


def test_versions_from_parentdir_2():
    """
    Test the NotThisMethod exception with non-existent directory
    """
    with pytest.raises(NotThisMethod):
        versions_from_parentdir("unknown", "", "")


def test_versions_from_parentdir_3(capsys):
    """
    Test verbose parameter
    """
    expect_string = "Tried directories"
    try:
        versions_from_parentdir("unknown", "", "1")
        # captured = capsys.readoutstd()
    except:
        assert capsys.readouterr().out.startswith(expect_string)
