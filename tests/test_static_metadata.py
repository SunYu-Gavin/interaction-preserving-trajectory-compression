import pickle
import pytest
from iptc.static_metadata import inspect_metadata, Node


def test_container_syntax_and_shared_reference():
    shared = [1, 2.5, True, None, b"x"]
    result = inspect_metadata(pickle.dumps({"a":shared,"b":shared},protocol=4))
    assert result["a"] == shared
    assert result["a"] is result["b"]


def forbidden_constructor():
    raise AssertionError("Reducer must never execute")


class Executable:
    def __reduce__(self):
        return forbidden_constructor, ()


def test_reducer_is_inert():
    result = inspect_metadata(pickle.dumps(Executable(),protocol=4))
    assert isinstance(result, Node)
    assert result.symbol.endswith(".forbidden_constructor")


def test_trailing_payload_rejected():
    with pytest.raises(ValueError,match="tail"):
        inspect_metadata(pickle.dumps([1],protocol=4)+b"extra")


def test_unsupported_opcode_rejected():
    with pytest.raises(ValueError,match="Unsupported"):
        inspect_metadata(b"Pexternal\n.")
