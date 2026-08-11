import re

import numpy as np
import pytest

import astropy.units as u

from sunpy.net import base_client, dataretriever, jsoc, vso
from sunpy.net.base_client import QueryResponseRow, QueryResponseTable, convert_row_to_table
from sunpy.net.dataretriever.sources.norh import NoRHClient

_REGEX = re.compile(r"Client")

CLIENT_LIST = []

for a_import in [vso, jsoc, dataretriever]:
    for item in dir(a_import):
        if _REGEX.search(item):
            CLIENT_LIST.append(getattr(a_import, item))

CLIENT_LIST.remove(dataretriever.client.GenericClient)

# We can access the registry directly
CLIENT_NAMES = base_client.BaseClient._registry.keys()
CLIENTS_REG = base_client.BaseClient._registry.items()


@pytest.mark.parametrize("client", CLIENT_LIST)
def test_registry(client):
    """
    Check if each client has been registered.
    """
    assert client in CLIENT_NAMES
    assert (client, client._can_handle_query) in CLIENTS_REG


@pytest.fixture
def dummy_response():
    return QueryResponseTable([{'hello': 1}], client=NoRHClient())


def test_slice(dummy_response):
    assert len(dummy_response) == 1

    row = dummy_response[0]
    table = row.as_table()

    assert len(table) == 1
    assert isinstance(table.client, NoRHClient)

    col = dummy_response['hello']
    table = col.as_table()

    assert len(table) == 1
    assert isinstance(table.client, NoRHClient)


def test_path_format_keys(dummy_response):
    assert dummy_response.path_format_keys() == {'hello'}


def test_path_format_keys_intersection(monkeypatch):
    """
    ``path_format_keys`` is documented to return only the keys common to
    every row. Simulate rows with differing ``response_block_map`` keys
    (e.g. mixed-provider results) and confirm the intersection is actually
    computed, rather than just returning the first row's keys.
    """
    table = QueryResponseTable([{'hello': 1}, {'hello': 2}], client=NoRHClient())

    # Keyed by the row's 'hello' value rather than row.index, since slicing
    # (self[1:] in path_format_keys) resets each sub-table's row indices
    # back to 0.
    per_row_keys = {1: {'hello', 'world'}, 2: {'hello'}}

    def fake_response_block_map(self):
        return dict.fromkeys(per_row_keys[self['hello']])

    monkeypatch.setattr(
        QueryResponseRow, 'response_block_map', property(fake_response_block_map)
    )

    assert table.path_format_keys() == {'hello'}


@pytest.mark.parametrize("size_bytes", [1, 500, 999])
def test_total_size_under_one_kilobyte(size_bytes):
    """
    Totals under 1000 bytes need no SI prefix. ``total_size`` used to crash
    with a ``KeyError`` because astropy's si_prefixes table has no entry
    for the "no prefix" value of 1.0.
    """
    table = QueryResponseTable([{'size': size_bytes}])
    table.size_column = 'size'
    table['size'] = table['size'] * u.byte

    result = table.total_size()

    assert result.unit == u.byte
    assert result.value == size_bytes


def test_total_size_kilobytes():
    table = QueryResponseTable([{'size': 2500}])
    table.size_column = 'size'
    table['size'] = table['size'] * u.byte

    result = table.total_size()

    assert result.unit == u.kilobyte
    np.testing.assert_allclose(result.value, 2.5)


def test_convert_row_to_table(dummy_response):

    @convert_row_to_table
    def example(self, query_results, **kwargs):
        return query_results

    assert example(None, dummy_response) is dummy_response
    # This is a single row table anyway
    assert example(None, dummy_response[0]) == dummy_response
