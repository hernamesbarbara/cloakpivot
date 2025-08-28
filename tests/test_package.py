"""Basic tests to validate package structure and imports."""

from importlib import import_module


def test_package_imports() -> None:
    """Test that all package modules can be imported."""
    # Test main package
    cloakpivot = import_module("cloakpivot")
    assert hasattr(cloakpivot, "__version__")
    assert cloakpivot.__version__ == "0.1.0"

    # Test submodules
    import_module("cloakpivot.cli")
    import_module("cloakpivot.core")
    import_module("cloakpivot.policies")
    import_module("cloakpivot.utils")


def test_cli_import() -> None:
    """Test that CLI module can be imported and has expected functions."""
    from cloakpivot.cli import cli, main
    assert callable(main)
    assert callable(cli)


def test_version_consistency() -> None:
    """Test that version is consistent across package."""
    import cloakpivot
    assert isinstance(cloakpivot.__version__, str)
    assert len(cloakpivot.__version__.split(".")) == 3
