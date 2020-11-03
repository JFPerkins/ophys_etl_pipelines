import numpy as np
import pandas as pd
import pytest

from ophys_etl.transforms import sine_dewarp


@pytest.fixture
def sample_video():
    return np.array(
        [
            [
                [1, 2, 3, 4, 5],
                [1, 6, 7, 8, 9],
                [1, 2, 7, 4, 9]
            ],
            [
                [1, 2, 3, 4, 5],
                [1, 6, 7, 8, 9],
                [1, 2, 7, 4, 9]
            ]
        ]
    )


@pytest.fixture
def random_sample_video():
    return np.random.randint(low=0, high=2, size=(20, 512, 512))


@pytest.fixture
def random_sample_xtable(random_sample_video):
    return sine_dewarp.create_xtable(
        movie=random_sample_video,
        aL=160.0, aR=160.0,
        bL=85.0, bR=90.0,
        noise_reduction=0
    )


def test_mode(sample_video):
    mode, count = sine_dewarp.mode(sample_video)

    assert 1 == len(mode)
    assert 1 in mode
    assert 6 == count


def test_mode_extra_value():
    img = np.array(
        [
            [1, 2],
            [1, 2],
            [1, 2]
        ]
    )

    mode, count = sine_dewarp.mode(img)

    assert 2 == len(mode)
    assert 1 in mode
    assert 2 in mode
    assert 3 == count


def test_xdewarp(random_sample_video, random_sample_xtable):
    frame, output = sine_dewarp.xdewarp(
        frame=0,
        data=random_sample_video,
        FOVwidth=512,
        xtable=random_sample_xtable,
        noise_reduction=0
    )

    assert 0 == frame
    assert random_sample_video[0].shape == output.shape