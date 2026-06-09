"""Tests for svZeroD binary path resolution (no solver build required)."""

import stat

import pytest

from util.zerod_calibration.tools import svzerod_binaries


def test_svzerod_install_dir_uses_env(monkeypatch, tmp_path):
    install = tmp_path / "custom_release"
    install.mkdir()
    monkeypatch.setenv("SVZEROD_INSTALL_DIR", str(install))
    assert svzerod_binaries.svzerod_install_dir() == str(install.resolve())


def test_svzerod_binary_found_in_install_dir(monkeypatch, tmp_path):
    install = tmp_path / "Release"
    install.mkdir()
    binary = install / "svzerodsolver"
    binary.write_text("#!/bin/sh\necho ok\n")
    binary.chmod(binary.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("SVZEROD_INSTALL_DIR", str(install))
    monkeypatch.delenv("PATH", raising=False)
    assert svzerod_binaries.svzerod_binary("svzerodsolver") == str(binary)


def test_svzerod_binary_found_on_path(monkeypatch, tmp_path):
    install = tmp_path / "empty_install"
    install.mkdir()
    binary = tmp_path / "bin" / "svzerodcalibrator"
    binary.parent.mkdir()
    binary.write_text("#!/bin/sh\necho ok\n")
    binary.chmod(binary.stat().st_mode | stat.S_IXUSR)
    # Empty install dir so resolution falls through to PATH (not the repo sibling default).
    monkeypatch.setenv("SVZEROD_INSTALL_DIR", str(install))
    monkeypatch.setenv("PATH", str(binary.parent))
    assert svzerod_binaries.svzerod_binary("svzerodcalibrator") == str(binary)


def test_svzerod_binary_missing_raises(monkeypatch, tmp_path):
    install = tmp_path / "empty"
    install.mkdir()
    monkeypatch.setenv("SVZEROD_INSTALL_DIR", str(install))
    monkeypatch.setenv("PATH", "")
    with pytest.raises(RuntimeError, match="svzerodsolver not found"):
        svzerod_binaries.svzerod_binary("svzerodsolver")
