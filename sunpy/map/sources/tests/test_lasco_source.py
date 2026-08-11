"""
Test cases for SOHO LASCOMap subclass.
"""
import copy

import numpy as np
import pytest

import astropy.units as u

from sunpy.data.test import get_dummy_map_from_header, get_test_filepath
from sunpy.map import Map
from sunpy.map.mapbase import SpatialPair
from sunpy.map.sources.soho import LASCOMap
from sunpy.tests.helpers import skip_glymur
from sunpy.util.exceptions import SunpyMetadataWarning
from .helpers import _test_private_date_setters

header_list = [
    "lasco_c2_25299383_s.header",
    "lasco_c3.header",
]


@pytest.fixture(scope="module", params=header_list, ids=['C2', 'C3'])
def lasco_map(request):
    return get_dummy_map_from_header(get_test_filepath(request.param))


@pytest.fixture
def lasco_helioviewer():
    return Map(get_test_filepath("2013_05_13__16_54_06_137__SOHO_LASCO_C3_white-light.jp2"))


def test_fitstoLASCO(lasco_map):
    """Tests the creation of LASCOMap using FITS."""
    assert isinstance(lasco_map, LASCOMap)


def test_lasco_coordinate_system(lasco_map):
    assert lasco_map.coordinate_system ==  SpatialPair(axis1='HPLN-TAN', axis2='HPLT-TAN')


def test_is_datasource_for(lasco_map):
    """Test the is_datasource_for method of LASCOMap.
    Note that header data to be provided as an argument
    can be a MetaDict object."""
    assert lasco_map.is_datasource_for(lasco_map.data, lasco_map.meta)


def test_measurement(lasco_map):
    """Tests the measurement property of the LASCOMap object."""
    assert lasco_map.measurement == "white-light"


def test_wavelength(lasco_map):
    """Tests wavelength property."""
    assert lasco_map.wavelength is None


def test_reference_date(lasco_map):
    assert lasco_map.reference_date.isot in ["2009-02-28T00:05:33.380", "2002-05-21T00:00:00.000"]


def test_date(lasco_map):
    assert lasco_map.date.isot in ["2009-02-28T00:05:33.380", "2002-05-21T00:18:06.516"]


def test_private_date_setters(lasco_map):
    _test_private_date_setters(lasco_map)


def test_nickname(lasco_map):
    assert lasco_map.nickname == {'C2': 'LASCO-C2 Orange',
                                  'C3': 'LASCO-C3 Clear'}[lasco_map.detector]

    with pytest.raises(AttributeError, match="Cannot manually set nickname for LASCOMap"):
        lasco_map.nickname = 'new nickname'


def test_observatory(lasco_map):
    """Tests the observatory property of the LASCOMap object."""
    assert lasco_map.observatory == "SOHO"


def test_norm_clip(lasco_map):
    # Tests that the default normalizer has clipping disabled
    assert not lasco_map.plot_settings['norm'].clip


@skip_glymur
def test_helioviewer_rotation(lasco_map, lasco_helioviewer):
    """Tests that rotation metadata is correctly removed
    for JPEG2000 images provided by Helioviewer.org."""
    rmatrix = {'C2': [[0.999966, -0.008296], [0.008296, 0.999966]],
               'C3': [[1, 0], [0, 1]]}[lasco_map.detector]
    np.testing.assert_allclose(lasco_map.rotation_matrix, rmatrix, rtol=1e-6)
    np.testing.assert_array_equal(lasco_helioviewer.rotation_matrix, [[1., 0.], [0., 1.]])


@skip_glymur
def test_lasco_helioviewer_unit(lasco_helioviewer):
    assert lasco_helioviewer.unit == u.dimensionless_unscaled


def test_wcs(lasco_map):
    # Smoke test that WCS is valid and can transform from pixels to world coordinates
    with pytest.warns(SunpyMetadataWarning, match='Missing metadata for observer'):
        lasco_map.pixel_to_world(0*u.pix, 0*u.pix)


def test_spatial_units_missing_cunit(lasco_map):
    # CUNIT1/CUNIT2 are optional FITS keywords. Missing them used to crash
    # with AttributeError: 'NoneType' object has no attribute 'lower'.
    m = copy.deepcopy(lasco_map)
    del m.meta['cunit1']
    del m.meta['cunit2']
    assert m.spatial_units == SpatialPair(axis1=u.arcsec, axis2=u.arcsec)


def test_set_date_no_time_key(lasco_map):
    # If neither time-obs nor time_obs is present at all, _set_date used to
    # raise UnboundLocalError because time_key was never assigned.
    m = copy.deepcopy(lasco_map)
    m.meta.pop('time-obs', None)
    m.meta.pop('time_obs', None)

    m._set_date("2012-03-04 05:06:07")

    assert m.date.isot == "2012-03-04T05:06:07.000"


def test_set_date_preserves_time_of_day(lasco_map):
    # When a time-obs key is present, _set_date deleted it and then checked
    # for its presence *after* deleting it, so the branch that splits the
    # date and time back into separate keys was unreachable, and the
    # time-obs key was silently dropped from the metadata.
    m = copy.deepcopy(lasco_map)
    m.meta['time-obs'] = 'placeholder'
    m.meta.pop('time_obs', None)
    assert 'date-obs' in m.meta

    m._set_date("2012-03-04 05:06:07")

    assert 'time-obs' in m.meta
    assert m.meta['time-obs'] == '05:06:07.000'
    assert 'T' not in m.meta['date-obs']
    assert m.meta['date-obs'] == '2012-03-04'
    assert m.date.isot == "2012-03-04T05:06:07.000"
