"""SAM-AASIST network: the vendored authors' AASIST with the SAM authors' pooling ratios.

The pinned SAM-AntiSpoofing `models/aasist.py` differs from our vendored clovaai
AASIST only in its hard-coded pooling ratios ([0.5, 0.5, 0.5, 0.5] rather than
[0.5, 0.7, 0.5, 0.5]) and its (logits, hidden) return order; both are checked
against `upstream/upstream_aasist.py` by the tests. Production AASIST is unchanged.
"""

from copy import deepcopy

from audio_deepfake_detection.sota.aasist import MODEL_CONFIG as AASIST_MODEL_CONFIG
from audio_deepfake_detection.sota.aasist import NUM_SAMPLES, UPSTREAM_COMMIT as AASIST_UPSTREAM_COMMIT
from audio_deepfake_detection.sota.aasist import _CachedSincConv
from audio_deepfake_detection.sota.aasist_arch import Model

SAM_UPSTREAM_REPOSITORY = "https://github.com/nii-yamagishilab/SAM-AntiSpoofing"
SAM_UPSTREAM_COMMIT = "59c69de3ffab8025acb815875f29e93a3069747f"
# Git blob SHA-1 values match the pinned tree; SHA-256 values bind the local byte copies.
UPSTREAM_FILES = {
    "LICENSE": ("65e5aa377a442758e1c0efec7ed77c2f8d9bb72d",
                "a792d1a07c322744a9f3f222c7189d0ba7fc501b705e82433d6784cffee08000"),
    "RawBoost.py": ("a908c3d8d9f45242b884aa200533f8ce6e567dcb",
                    "d1021d7d0c020a4e09c4dfda993152112ced642cbc29e28286740d1e6937b12c"),
    "models/aasist.py": ("5865074c156685d5048f55a075915ec3ba5f71c5",
                         "c3e45d58e70ac99ac37fd1e2d484e423f9eb171fbefc34c3c75bba4887a3c82a"),
    "optimizers/sam.py": ("0a7b999343e13da152af1a556610ba97f492499f",
                          "fc0756a63e57794013fe930a6530f42e386061fbe7c416b9c0aefed89cd72997"),
    "optimizers/bypass_bn.py": ("89a72b094c841b94db2e397d67839442175b81cc",
                                "4e5a60b235952fdc6c5b2d4ba974ddf5d9d61d7d6d8edcc9d91d7c5e11be8b57"),
}
MODEL_CONFIG = {**deepcopy(AASIST_MODEL_CONFIG), "pool_ratios": [0.5, 0.5, 0.5, 0.5]}
POOL_RATIO_STATUS = ("released SAM architecture: SAM authors' models/aasist.py uses 0.5 for the "
                     "temporal GraphPool; the project's AASIST baseline keeps the clovaai 0.7")
PARAMETER_COUNT = 297866


class SAMAASIST(Model):
    """Random initialization; no pretrained or released anti-spoofing weights."""

    def __init__(self, *, cache_sinc_filters=True):
        super().__init__(deepcopy(MODEL_CONFIG))
        if cache_sinc_filters:
            # Fixed sinc filters stay on-device; parameter and checkpoint keys are unchanged.
            self.conv_time = _CachedSincConv(out_channels=MODEL_CONFIG["filts"][0],
                                             kernel_size=MODEL_CONFIG["first_conv"], in_channels=1)

    def forward(self, waveforms):
        if waveforms.ndim != 2 or waveforms.shape[1] != NUM_SAMPLES:
            raise ValueError(f"Expected [batch, {NUM_SAMPLES}] waveforms")
        # The SAM configs never enable frequency masking; RawBoost belongs to the data path.
        _, original_logits = super().forward(waveforms, Freq_aug=False)
        # Upstream order is [spoof, bona fide]; reverse on-device to public [bona fide, spoof].
        return original_logits.flip(1)

    @staticmethod
    def spoof_score(logits):
        return logits[:, 1] - logits[:, 0]


__all__ = ["AASIST_UPSTREAM_COMMIT", "MODEL_CONFIG", "PARAMETER_COUNT", "POOL_RATIO_STATUS",
           "SAMAASIST", "SAM_UPSTREAM_COMMIT", "SAM_UPSTREAM_REPOSITORY", "UPSTREAM_FILES"]
