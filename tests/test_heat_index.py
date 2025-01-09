import numpy as np
import pytest

from pythermalcomfort.classes_input import HIModels
from pythermalcomfort.models import heat_index
from tests.conftest import Urls, retrieve_reference_table, validate_result


def test_heat_index(get_test_url, retrieve_data):
    reference_table = retrieve_reference_table(
        get_test_url, retrieve_data, Urls.HEAT_INDEX.name
    )
    tolerance = reference_table["tolerance"]

    for entry in reference_table["data"]:
        inputs = entry["inputs"]
        outputs = entry["outputs"]
        result = heat_index(**inputs, model=HIModels.rothfusz.value)

        validate_result(result, outputs, tolerance)


def test_extended_heat_index():
    index = 0
    hi_test_values = [
        199.9994020652,
        199.9997010342,
        200.0000000021,
        209.9975943902,
        209.9987971085,
        209.9999998068,
        219.9915822029,
        219.9957912306,
        219.9999999912,
        229.9739691979,
        229.9869861009,
        230.0000001850,
        239.9253828022,
        239.9626700074,
        240.0000000003,
        249.7676757244,
        249.8837049107,
        250.0000000037,
        259.3735990024,
        259.6864068902,
        259.9999999944,
        268.5453870455,
        269.2745889562,
        270.0000002224,
        277.2234200026,
        278.6369451963,
        280.0000000091,
        285.7510545370,
        288.2813660100,
        290.7860610129,
        297.5737503539,
        300.2922595865,
        305.3947127590,
        305.5549530893,
        318.6225524695,
        359.9063248191,
        313.0298872791,
        359.0538750602,
        407.5345212438,
        320.5088548469,
        398.5759733823,
        464.9949352940,
        328.0358006469,
        445.8599463105,
        530.5524786708,
        333.2806160592,
        500.0421800191,
        601.9518435268,
        343.6312984164,
        559.6640227151,
        677.2462089759,
        354.1825692377,
        623.1960299857,
        755.0832658147,
    ]
    for t in range(200, 380, 10):
        for rh in [0, 0.5, 1]:
            hi = heat_index(t - 273.15, rh * 100, model=HIModels.lu_romps.value).hi
            assert np.isclose(hi + 273.15, hi_test_values[index], atol=1)
            index += 1


def test_extended_heat_index_array_input():
    hi = heat_index([20, 40], 50, model=HIModels.lu_romps.value).hi
    assert np.allclose(hi, [19.0, 63.4], atol=0.1)


def test_wrong_model():
    with pytest.raises(ValueError):
        heat_index(tdb=25, rh=50, model="random")
