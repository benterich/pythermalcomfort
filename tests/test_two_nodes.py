from pythermalcomfort.models import gagge_two_nodes
from tests.conftest import Urls, retrieve_reference_table, validate_result


def test_two_nodes(get_test_url, retrieve_data):
    reference_table = retrieve_reference_table(
        get_test_url, retrieve_data, Urls.TWO_NODES.name
    )
    tolerance = reference_table["tolerance"]

    for entry in reference_table["data"]:
        inputs = entry["inputs"]
        outputs = entry["outputs"]
        result = gagge_two_nodes(**inputs)

        validate_result(result, outputs, tolerance)
