import pytest

from bert_optimization import glue_processor as P


def test_read_table_csv(tmpdir):
    tmpdir.join("test.csv").write("\n".join(["1,2,3,4", "4,5,6,7"]))

    lines = P.read_table(tmpdir.join("test.csv"), ",")
    assert len(lines) == 2
    assert lines == [["1", "2", "3", "4"], ["4", "5", "6", "7"]]


def test_read_table_tsv(tmpdir):
    tmpdir.join("test.tsv").write("\n".join(["1\t2\t3\t4", "4\t5\t6\t7"]))

    lines = P.read_table(tmpdir.join("test.tsv"))
    assert len(lines) == 2
    assert lines == [["1", "2", "3", "4"], ["4", "5", "6", "7"]]


@pytest.mark.parametrize(
    "input_row, is_test, expected_result",
    [
        pytest.param(
            [
                ["", "1", "", "The sailors rode the breeze clear of the rocks."],
                ["", "1", "", "The weights made the rope stretch over the pulley."],
            ],
            False,
            (
                ["1", "1"],
                [
                    "The sailors rode the breeze clear of the rocks.",
                    "The weights made the rope stretch over the pulley.",
                ],
            ),
        ),
        pytest.param(
            [
                ["Dummy"],
                ["1", "The sailors rode the breeze clear of the rocks."],
                ["1", "The weights made the rope stretch over the pulley."],
            ],
            True,
            (
                None,
                [
                    "The sailors rode the breeze clear of the rocks.",
                    "The weights made the rope stretch over the pulley.",
                ],
            ),
        ),
    ],
)
def test_parse_cola_dataset(input_row, is_test, expected_result):
    assert expected_result == P.CoLAProcessor.parse_cola_dataset(input_row, is_test)
