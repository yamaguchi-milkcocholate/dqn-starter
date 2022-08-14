import pytest
import sys
from pathlib import Path

rootdir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(rootdir))
import src.markets.ppocat as markets


@pytest.fixture
def action_parser_2_002() -> markets.ActionParser:
    return markets.ActionParser(num_discrete=2, max_spread=0.02)


@pytest.fixture
def action_parser_3_003() -> markets.ActionParser:
    return markets.ActionParser(num_discrete=3, max_spread=0.03)


class Test_ActionParserクラスはカテゴリ変数から行動に変換する:
    class Test_カテゴリ変数とサイドとスプレッドの対応表を持つ:
        def test_指値数2_スプレッド002の場合(self, action_parser_2_002: markets.ActionParser):
            expected = {
                0: ("Buy", -0.02),
                1: ("Buy", -0.01),
                2: ("Buy", 0.00),
                3: ("Buy", 0.01),
                4: ("Buy", 0.02),
                5: ("Sell", -0.02),
                6: ("Sell", -0.01),
                7: ("Sell", 0.00),
                8: ("Sell", 0.01),
                9: ("Sell", 0.02),
                10: ("Hold", None),
                11: ("Cancel", None),
            }
            actual = action_parser_2_002.cat_action_dict
            assert len(expected) == len(actual)
            for _expected, _actual in zip(expected.items(), actual.items()):
                assert _expected[0] == _actual[0]
                assert _expected[1][0] == _actual[1][0]
                assert _expected[1][1] == pytest.approx(_actual[1][1], 0.01)

        def test_指値数3_スプレッド003の場合(self, action_parser_3_003: markets.ActionParser):
            expected = {
                0: ("Buy", -0.03),
                1: ("Buy", -0.02),
                2: ("Buy", -0.01),
                3: ("Buy", 0.00),
                4: ("Buy", 0.01),
                5: ("Buy", 0.02),
                6: ("Buy", 0.03),
                7: ("Sell", -0.03),
                8: ("Sell", -0.02),
                9: ("Sell", -0.01),
                10: ("Sell", 0.00),
                11: ("Sell", 0.01),
                12: ("Sell", 0.02),
                13: ("Sell", 0.03),
                14: ("Hold", None),
                15: ("Cancel", None),
            }
            actual = action_parser_3_003.cat_action_dict
            assert len(expected) == len(actual)
            for _expected, _actual in zip(expected.items(), actual.items()):
                assert _expected[0] == _actual[0]
                assert _expected[1][0] == _actual[1][0]
                assert _expected[1][1] == pytest.approx(_actual[1][1], 0.01)
