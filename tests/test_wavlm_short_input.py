from __future__ import annotations

from humomni.features.wavlm_extract import _minimum_wavlm_input_samples


def test_wavlm_minimum_input_matches_standard_conv_frontend() -> None:
    class Config:
        conv_kernel = [10, 3, 3, 3, 3, 2, 2]
        conv_stride = [5, 2, 2, 2, 2, 2, 2]

    class Model:
        config = Config()

    assert _minimum_wavlm_input_samples(Model()) == 400
