import numpy as np
import pytest

from guideloom.shuffle import FeistelShuffler


@pytest.mark.parametrize("index", [2**63, 2**64 - 1])
def test_feistel_rejects_unsupported_index(index: int) -> None:
    shuffler = FeistelShuffler.from_seed(2**63 - 1, seed=0)
    with pytest.raises(IndexError):
        shuffler.shuffle(index)
    with pytest.raises(IndexError):
        shuffler.unshuffle(index)


@pytest.mark.parametrize("length", [10, 1_000, 1_000_000, 1_000_000_000, 2**63 - 1])
@pytest.mark.parametrize("seed", [0, 123, 431237, 12312321, 1233215161, 2**32 - 1])
def test_feistel_inversion(length: int, seed: int) -> None:
    indices = np.random.default_rng(seed).integers(length, size=10000, dtype=np.uint64).tolist()
    shuffler = FeistelShuffler.from_seed(length, seed=seed, rounds=5)
    for index in indices:
        assert shuffler.unshuffle(shuffler.shuffle(index)) == index


@pytest.mark.parametrize("seed", [0, 123, 431237, 12312321, 1233215161, 2**32 - 1])
def test_feistel_inversion_whole_range(seed: int) -> None:
    shuffler = FeistelShuffler.from_seed(2**63 - 1, seed=seed, rounds=5)
    indices = [2**p - 1 for p in range(63)]
    for index in indices:
        assert shuffler.unshuffle(shuffler.shuffle(index)) == index


@pytest.mark.parametrize("seed", [0, 123, 431237, 12312321, 1233215161, 2**32 - 1])
def test_feistel_inversion_whole_large_numbers(seed: int) -> None:
    shuffler = FeistelShuffler.from_seed(2**63 - 1, seed=seed, rounds=5)
    indices = [2**63 - 2**p - 1 for p in range(63)]
    for index in indices:
        assert shuffler.unshuffle(shuffler.shuffle(index)) == index


@pytest.fixture
def hardcoded() -> tuple[list[int], list[int]]:
    indices = [
        766884,
        834981,
        124178,
        596554,
        80263,
        288863,
        443470,
        42951,
        587852,
        973654,
        947227,
        596471,
        411878,
        790263,
        924844,
        910339,
        3491,
        688154,
        719125,
        189991,
        505527,
        981478,
        687205,
        284740,
        322427,
        629273,
        144629,
        581036,
        572131,
        599912,
        774252,
        535248,
        724262,
        995777,
        128543,
        501946,
        126431,
        771022,
        141781,
        494161,
        227929,
        997674,
        430941,
        978621,
        882632,
        393568,
        448726,
        321925,
        801162,
        862180,
        488259,
        799324,
        963507,
        691430,
        704416,
        408517,
        965294,
        389765,
        643907,
        131648,
        631718,
        625497,
        179581,
        82401,
        829474,
        274572,
        840426,
        656117,
        110855,
        14676,
        126247,
        834532,
        478682,
        72399,
        229735,
        524120,
        815111,
        546550,
        626126,
        227246,
        821159,
        955978,
        411743,
        340963,
        885622,
        504707,
        696167,
        867383,
        403714,
        656952,
        569805,
        384722,
        280952,
        96272,
        142693,
        955805,
        967193,
        589567,
        33649,
        582162,
    ]
    expectations = [
        163701,
        390033,
        690210,
        238824,
        205424,
        122519,
        763577,
        797869,
        316704,
        488496,
        653427,
        92980,
        760936,
        554449,
        256522,
        343552,
        624827,
        664371,
        100975,
        227389,
        976050,
        286841,
        963738,
        181329,
        565445,
        839249,
        491312,
        115780,
        226902,
        968922,
        563567,
        820225,
        703420,
        336833,
        4694,
        970195,
        50713,
        244800,
        245541,
        276569,
        811175,
        830863,
        5560,
        366025,
        14759,
        644055,
        524421,
        202141,
        353273,
        39741,
        170859,
        905326,
        671443,
        562549,
        443992,
        84669,
        517169,
        148327,
        42131,
        968801,
        106813,
        989397,
        767885,
        623312,
        782312,
        309655,
        682479,
        237008,
        389886,
        895142,
        47586,
        388232,
        287368,
        651813,
        485729,
        146122,
        907672,
        94377,
        60885,
        503279,
        694621,
        127907,
        385726,
        352483,
        986390,
        736863,
        108266,
        166771,
        620375,
        871025,
        282747,
        185257,
        56403,
        179062,
        60463,
        679739,
        669241,
        300192,
        40925,
        843996,
    ]
    return (indices, expectations)


def test_feistel_hardcoded(hardcoded: tuple[list[int], list[int]]) -> None:
    seed = 100
    length = 1_000_000

    indices, expectations = hardcoded
    shuffler = FeistelShuffler.from_seed(length, seed=seed, rounds=5)
    for index, expected in zip(indices, expectations, strict=True):
        assert shuffler.shuffle(index) == expected
        assert shuffler.unshuffle(expected) == index


@pytest.mark.parametrize("length", [10, 100, 200, 500, 1_000, 2_500, 5_000, 10_000, 100_000])
@pytest.mark.parametrize("seed", [0, 123, 431237, 12312321, 1233215161, 2**32 - 1])
def test_feistel_nonreplacement_permutation(length: int, seed: int) -> None:
    shuffler = FeistelShuffler.from_seed(length, seed=seed, rounds=5)
    indices = np.array([shuffler.shuffle(i) for i in range(length)], dtype=np.uint64)
    np.testing.assert_equal(np.sort(indices), np.arange(length))
