from pythermalcomfort.models import adaptive_en
import pytest
from tests.conftest import Urls, retrieve_reference_table, validate_result


def test_adaptive_en(get_test_url, retrieve_data):
    reference_table = retrieve_reference_table(
        get_test_url, retrieve_data, Urls.ADAPTIVE_EN.name
    )
    tolerance = reference_table["tolerance"]

    for entry in reference_table["data"]:
        inputs = entry["inputs"]
        outputs = entry["outputs"]
        result = adaptive_en(
            inputs["tdb"], inputs["tr"], inputs["t_running_mean"], inputs["v"]
        )

        validate_result(result, outputs, tolerance)


def test_ashrae_inputs_invalid_units():
    with pytest.raises(
        ValueError, match=r"Invalid unit: INVALID. Supported units are \['SI', 'IP'\]."
    ):
        adaptive_en(tdb=25, tr=25, t_running_mean=20, v=0.1, units="INVALID")


def test_ashrae_inputs_invalid_tdb_type():
    with pytest.raises(TypeError, match="tdb must be a float, int, or array-like."):
        adaptive_en(tdb="invalid", tr=25, t_running_mean=20, v=0.1)


def test_ashrae_inputs_invalid_tr_type():
    with pytest.raises(TypeError, match="tr must be a float, int, or array-like."):
        adaptive_en(tdb=25, tr="invalid", t_running_mean=20, v=0.1)


def test_ashrae_inputs_invalid_t_running_mean_type():
    with pytest.raises(
        TypeError, match="t_running_mean must be a float, int, or array-like."
    ):
        adaptive_en(tdb=25, tr=25, t_running_mean="invalid", v=0.1)


def test_ashrae_inputs_invalid_v_type():
    with pytest.raises(TypeError, match="v must be a float, int, or array-like."):
        adaptive_en(tdb=25, tr=25, t_running_mean=20, v="invalid")
