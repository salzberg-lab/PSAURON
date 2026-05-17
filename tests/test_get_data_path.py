import os

from psauron.psauron import get_data_path


def test_get_data_path_returns_string():
    path = get_data_path()
    assert isinstance(path, str)


def test_get_data_path_file_exists():
    path = get_data_path()
    assert os.path.isfile(path), f"Model file not found at {path}"


def test_get_data_path_correct_filename():
    path = get_data_path()
    assert path.endswith("model_state_dict.pt")


def test_get_data_path_nonzero_size():
    path = get_data_path()
    assert os.path.getsize(path) > 0, "Model file is empty"


def test_get_data_path_inside_package():
    """Verify the model file is resolved within the psauron package directory."""
    path = get_data_path()
    parts = os.path.normpath(path).split(os.sep)
    assert "psauron" in parts, f"Path does not appear to be inside the psauron package: {path}"
    assert "data" in parts, f"Path does not include expected 'data' subdirectory: {path}"


def test_no_pkg_resources_dependency():
    """Confirm get_data_path works without pkg_resources being imported."""
    import sys
    # Temporarily hide pkg_resources from the import system
    saved = sys.modules.get("pkg_resources")
    sys.modules["pkg_resources"] = None
    try:
        path = get_data_path()
        assert os.path.isfile(path)
    finally:
        if saved is not None:
            sys.modules["pkg_resources"] = saved
        else:
            sys.modules.pop("pkg_resources", None)
