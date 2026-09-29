from ojev.cli import softmax, validate


def test_softmax_sums_to_one():
    values = softmax([1.0, 2.0, 3.0])
    assert abs(sum(values) - 1.0) < 1e-9
    assert values[2] > values[1] > values[0]


def test_validate():
    validate({
        "id": "x",
        "state": "evidence",
        "question": "criterion",
        "options": [
            {"id": "a", "description": "A"},
            {"id": "b", "description": "B"},
        ],
    })
