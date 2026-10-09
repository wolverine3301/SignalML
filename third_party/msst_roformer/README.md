# msst_roformer (vendored)

The BS-RoFormer and Mel-band RoFormer model definitions from ZFTurbo's
Music-Source-Separation-Training, vendored so `signalml/restore/roformer.py` can load the
community RoFormer checkpoints (vocal separation for S3; de-reverb, de-echo, denoise,
karaoke for restoration) without installing the whole training repo.

- Source: https://github.com/ZFTurbo/Music-Source-Separation-Training
- Commit: `84b1eac0887756b4f1a9d7a1ff49105939749ed2` (2026-09-26)
- Files: `models/bs_roformer/{bs_roformer,mel_band_roformer,attend}.py`, `LICENSE` (MIT)
- Change: one line in each model file, `from models.bs_roformer.attend import Attend` ->
  `from .attend import Attend`. Nothing else is edited; re-vendor rather than patch.
- Runtime deps (the `restore` extra): `einops`, `rotary-embedding-torch`, `beartype`,
  `packaging`, torch; the Mel-band model also uses `librosa` (a base dependency).

Checkpoints are not vendored. They live under `DATA_ROOT/models/<name>/` (see
`configs/dereverb.yaml`, `configs/separate.*.yaml`); each one's source, revision and
sha256 are recorded there. Only plain state dicts are accepted
(`torch.load(weights_only=True)`).
