"""S7 train-wrapper contract tests — offline, CPU, no trainer required.

The wrapper's whole job is to refuse a run that would only fail hours in, and to
leave behind enough provenance to say what a checkpoint came from. Both are tested
against a fake trainer tree: the vendored scripts and venv are never executed.
"""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

import pytest
import yaml

from signalml.config import active_profile
from signalml.train import status
from signalml.train.runner import (
    TrainConfig,
    execute,
    load_train_config,
    plan_run,
    preflight,
    write_run_record,
)

PROD = active_profile("prod")


def _trainer_tree(tmp_path, *, venv=True, vocoder=False):
    """A stand-in for third_party/DiffSinger: the paths preflight looks at."""
    d = tmp_path / "trainer"
    (d / "scripts").mkdir(parents=True)
    (d / "scripts" / "train.py").write_text("# fake", encoding="utf-8")
    (d / "scripts" / "binarize.py").write_text("# fake", encoding="utf-8")
    if venv:
        (d / ".venv" / "Scripts").mkdir(parents=True)
        (d / ".venv" / "Scripts" / "python.exe").write_text("", encoding="utf-8")
    if vocoder:
        ckpt = d / "checkpoints" / "pc_nsf_hifigan_44.1k_hop512_128bin_2025.02"
        ckpt.mkdir(parents=True)
        (ckpt / "model.ckpt").write_text("", encoding="utf-8")
    return d


def _dataset(tmp_path, name="ds1", *, val_with_vocoder=True, sample_rate=None):
    """A stand-in for a built dataset: one speaker folder + generated config."""
    root = tmp_path / "dr"
    ds = root / "datasets" / name
    (ds / "alice-en" / "wavs").mkdir(parents=True)
    (ds / "dictionary_en.txt").write_text("a\ta\n", encoding="utf-8")
    (ds / "dataset_card.md").write_text("# card\n", encoding="utf-8")
    config = {
        "base_config": ["configs/acoustic.yaml"],
        "dictionaries": {"en": str(ds / "dictionary_en.txt")},
        "datasets": [{"raw_data_dir": str(ds / "alice-en"), "speaker": "alice",
                      "spk_id": 0, "language": "en"}],
        "binary_data_dir": str(ds / "binary"),
        "num_spk": 1,
        "audio_sample_rate": sample_rate or PROD.sample_rate,
        "audio_num_mel_bins": PROD.n_mels,
        "hop_size": PROD.hop_length,
        "fft_size": PROD.n_fft,
        "win_size": PROD.win_length,
        "fmax": PROD.fmax,
        "mel_base": "e",
        "vocoder": "NsfHifiGAN",
        "vocoder_ckpt":
            "checkpoints/pc_nsf_hifigan_44.1k_hop512_128bin_2025.02/model.ckpt",
        "val_with_vocoder": val_with_vocoder,
    }
    (ds / "config_acoustic.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    return root, ds


def _cfg(trainer_dir) -> TrainConfig:
    return TrainConfig(trainer_dir=str(trainer_dir),
                       trainer_python=str(trainer_dir / ".venv" / "Scripts" / "python.exe"))


class TestConfig:
    def test_defaults_when_yaml_absent(self, tmp_path):
        cfg = load_train_config(tmp_path / "nope.yaml")
        assert cfg.trainer_dir == "third_party/DiffSinger"
        # IPA must survive Windows' cp1252 default all the way into the trainer
        assert cfg.env["PYTHONUTF8"] == "1"

    def test_repo_config_is_loadable(self):
        cfg = load_train_config()
        assert cfg.resolved_trainer_dir().name == "DiffSinger"

    def test_autodetects_venv_interpreter(self, tmp_path):
        trainer = _trainer_tree(tmp_path)
        cfg = TrainConfig(trainer_dir=str(trainer))
        assert cfg.resolved_trainer_python().exists()


class TestPlan:
    def test_commands_and_run_dir(self, tmp_path):
        trainer = _trainer_tree(tmp_path)
        root, ds = _dataset(tmp_path)
        plan = plan_run(root, "ds1", cfg=_cfg(trainer),
                        now=_dt.datetime(2026, 9, 20, 13, 45, 0))
        assert plan.exp_name == "ds1"  # defaults to the dataset name
        assert plan.cwd == trainer  # their configs' relative paths need this cwd
        assert plan.binarize_cmd[1:] == ["scripts/binarize.py", "--config",
                                         str(ds / "config_acoustic.yaml")]
        assert plan.train_cmd[1:] == ["scripts/train.py", "--config",
                                      str(ds / "config_acoustic.yaml"),
                                      "--exp_name", "ds1"]
        assert plan.run_dir == root / "runs" / "ds1" / "20260920_134500"
        assert plan.ckpt_dir == trainer / "checkpoints" / "ds1"

    def test_reset_and_hparams_passthrough(self, tmp_path):
        trainer = _trainer_tree(tmp_path)
        root, _ = _dataset(tmp_path)
        plan = plan_run(root, "ds1", cfg=_cfg(trainer), reset=True,
                        hparams="max_batch_size=8", exp_name="custom")
        assert "--reset" in plan.train_cmd
        assert plan.train_cmd[-2:] == ["--hparams", "max_batch_size=8"]
        assert plan.exp_name == "custom"

    def test_binarize_skipped_when_already_built(self, tmp_path):
        trainer = _trainer_tree(tmp_path)
        root, ds = _dataset(tmp_path)
        (ds / "binary").mkdir()
        (ds / "binary" / "train.data").write_text("", encoding="utf-8")
        plan = plan_run(root, "ds1", cfg=_cfg(trainer))
        assert plan.binarize_cmd is None
        assert any("reusing existing binary" in n for n in plan.notes)
        forced = plan_run(root, "ds1", cfg=_cfg(trainer), binarize=True)
        assert forced.binarize_cmd is not None

    def test_skipping_binarize_without_data_is_flagged(self, tmp_path):
        trainer = _trainer_tree(tmp_path)
        root, _ = _dataset(tmp_path)
        plan = plan_run(root, "ds1", cfg=_cfg(trainer), binarize=False)
        assert any("will fail immediately" in n for n in plan.notes)

    def test_vocoder_is_blocked_with_a_reason(self, tmp_path):
        trainer = _trainer_tree(tmp_path)
        root, _ = _dataset(tmp_path)
        with pytest.raises(NotImplementedError, match="SingingVocoders"):
            plan_run(root, "ds1", trainer="vocoder", cfg=_cfg(trainer))

    def test_variance_uses_its_own_config_and_says_when_it_is_missing(self, tmp_path):
        """D1 landed: variance is no longer refused outright, it just needs the
        config `dataset variance-config` writes."""
        trainer = _trainer_tree(tmp_path)
        root, ds = _dataset(tmp_path)
        cfg = _cfg(trainer)
        plan = plan_run(root, "ds1", trainer="variance", cfg=cfg)
        assert plan.config_path == ds / "config_variance.yaml"
        problems = preflight(plan, cfg)
        assert any("dataset build" in p for p in problems)


class TestPreflight:
    def test_clean_run_has_no_problems(self, tmp_path):
        trainer = _trainer_tree(tmp_path, vocoder=True)
        root, _ = _dataset(tmp_path)
        cfg = _cfg(trainer)
        assert preflight(plan_run(root, "ds1", cfg=cfg), cfg) == []

    def test_missing_vocoder_ckpt_blocks_before_the_gpu_starts(self, tmp_path):
        trainer = _trainer_tree(tmp_path, vocoder=False)
        root, _ = _dataset(tmp_path, val_with_vocoder=True)
        cfg = _cfg(trainer)
        problems = preflight(plan_run(root, "ds1", cfg=cfg), cfg)
        assert any("val_with_vocoder" in p for p in problems)

    def test_vocoder_ckpt_irrelevant_when_validation_is_off(self, tmp_path):
        trainer = _trainer_tree(tmp_path, vocoder=False)
        root, _ = _dataset(tmp_path, val_with_vocoder=False)
        cfg = _cfg(trainer)
        assert preflight(plan_run(root, "ds1", cfg=cfg), cfg) == []

    def test_unbuilt_dataset_says_how_to_build_it(self, tmp_path):
        trainer = _trainer_tree(tmp_path)
        cfg = _cfg(trainer)
        plan = plan_run(tmp_path / "dr", "nothing", cfg=cfg)
        problems = preflight(plan, cfg)
        assert len(problems) == 1 and "dataset build" in problems[0]

    def test_missing_trainer_venv_points_at_bootstrap(self, tmp_path):
        trainer = _trainer_tree(tmp_path, venv=False, vocoder=True)
        root, _ = _dataset(tmp_path)
        cfg = TrainConfig(trainer_dir=str(trainer))
        problems = preflight(plan_run(root, "ds1", cfg=cfg), cfg)
        assert any("bootstrap_rig" in p for p in problems)

    def test_foreign_dataset_paths_are_caught(self, tmp_path):
        """A config built on another machine carries its absolute paths."""
        trainer = _trainer_tree(tmp_path, vocoder=True)
        root, ds = _dataset(tmp_path)
        config = yaml.safe_load((ds / "config_acoustic.yaml").read_text(encoding="utf-8"))
        config["datasets"][0]["raw_data_dir"] = "Y:/elsewhere/alice-en"
        (ds / "config_acoustic.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
        cfg = _cfg(trainer)
        problems = preflight(plan_run(root, "ds1", cfg=cfg), cfg)
        assert any("absolute paths from the machine that built it" in p
                   for p in problems)

    def test_off_contract_mel_params_are_refused(self, tmp_path):
        """D5: a dataset binarized under parameters no profile describes."""
        trainer = _trainer_tree(tmp_path, vocoder=True)
        root, _ = _dataset(tmp_path, sample_rate=32000)
        cfg = _cfg(trainer)
        problems = preflight(plan_run(root, "ds1", cfg=cfg), cfg)
        assert any("matches no profile" in p for p in problems)


class TestRunRecord:
    def test_snapshots_and_provenance(self, tmp_path):
        trainer = _trainer_tree(tmp_path, vocoder=True)
        root, _ = _dataset(tmp_path)
        plan = plan_run(root, "ds1", cfg=_cfg(trainer))
        path = write_run_record(plan, probe=False)
        record = json.loads(path.read_text(encoding="utf-8"))

        assert record["exp_name"] == "ds1" and record["dataset"] == "ds1"
        assert record["audio"]["audio_sample_rate"] == PROD.sample_rate
        assert record["commands"]["train"] == plan.train_cmd
        assert "CC BY-NC" in record["vocoder"]["license"]
        # the inputs are copied next to the record, not merely referenced
        for name in ("config_acoustic.yaml", "dictionary_en.txt", "dataset_card.md"):
            assert (plan.run_dir / name).exists()
            assert name in record["snapshots_sha256"]
        assert "commit" in record["git"] or "error" in record["git"]


class TestExecute:
    def test_runs_binarize_then_train(self, tmp_path):
        trainer = _trainer_tree(tmp_path, vocoder=True)
        root, _ = _dataset(tmp_path)
        plan = plan_run(root, "ds1", cfg=_cfg(trainer))
        calls = []

        class Done:
            returncode = 0

        def fake(cmd, **kwargs):
            calls.append((cmd, kwargs["cwd"], kwargs["env"]["PYTHONUTF8"]))
            return Done()

        assert execute(plan, runner=fake) == 0
        assert [c[0][1] for c in calls] == ["scripts/binarize.py", "scripts/train.py"]
        assert calls[0][1] == str(trainer) and calls[0][2] == "1"

    def test_stops_when_binarize_fails(self, tmp_path):
        trainer = _trainer_tree(tmp_path, vocoder=True)
        root, _ = _dataset(tmp_path)
        plan = plan_run(root, "ds1", cfg=_cfg(trainer))
        calls = []

        class Failed:
            returncode = 3

        def fake(cmd, **kwargs):
            calls.append(cmd)
            return Failed()

        assert execute(plan, runner=fake) == 3
        assert len(calls) == 1  # train never launched


class TestStatus:
    """`train status` answers "where is it" and "is the good checkpoint still there".

    The probe runs in the trainer venv, so the subprocess is injected: these tests
    assert how we call it and what we make of its answer, never tensorboard itself.
    """

    def _probe(self, **over):
        probe = {
            "exp_dir": "/exp",
            "event_file": "events.out.tfevents.1",
            "event_age_sec": 12,
            "checkpoints": [
                {"name": "model_ckpt_steps_3000.ckpt", "step": 3000, "bytes": 10 ** 9},
                {"name": "model_ckpt_steps_4000.ckpt", "step": 4000, "bytes": 10 ** 9},
            ],
            "scalars": {
                "training/mel_loss": [[1000, 0.9, 100.0], [4000, 0.3, 400.0]],
                "validation/total_loss": [[1000, 0.6, 110.0], [3000, 0.2, 300.0],
                                          [4000, 0.5, 400.0]],
            },
            "audio": [],
        }
        probe.update(over)
        return probe

    def _runner(self, probe, calls=None):
        class Done:
            returncode = 0
            stdout = json.dumps(probe)
            stderr = ""

        def fake(cmd, **kwargs):
            if calls is not None:
                calls.append((cmd, kwargs))
            return Done()

        return fake

    def test_reports_step_rate_and_best_validation(self, tmp_path):
        trainer = _trainer_tree(tmp_path)
        s = status.collect("ds1", cfg=_cfg(trainer),
                           runner=self._runner(self._probe()))
        assert s["step"] == 4000
        assert s["steps_per_sec"] == 10.0  # 3000 steps over 300 wall seconds
        # the minimum, not the last value — a run that overfits ends above its best
        assert s["best"]["step"] == 3000 and s["best"]["value"] == 0.2
        assert s["best"]["latest_value"] == 0.5
        assert s["best"]["checkpoint_present"] is True

    def test_flags_a_best_checkpoint_that_was_rolled_off(self, tmp_path):
        """The 2026-09-20 case: val bottomed at 3000, num_ckpt_keep kept 19-20k."""
        trainer = _trainer_tree(tmp_path)
        probe = self._probe(checkpoints=[
            {"name": "model_ckpt_steps_4000.ckpt", "step": 4000, "bytes": 10 ** 9}])
        s = status.collect("ds1", cfg=_cfg(trainer), runner=self._runner(probe))
        assert s["best"]["checkpoint_present"] is False
        assert "rolling window" in status.format_status(s)

    def test_no_warning_before_the_first_checkpoint_exists(self, tmp_path):
        trainer = _trainer_tree(tmp_path)
        probe = self._probe(checkpoints=[])
        s = status.collect("ds1", cfg=_cfg(trainer), runner=self._runner(probe))
        assert "rolling window" not in status.format_status(s)

    def test_falls_back_to_mel_loss_when_total_is_absent(self, tmp_path):
        trainer = _trainer_tree(tmp_path)
        probe = self._probe(scalars={"validation/mel_loss": [[500, 0.3, 50.0],
                                                             [1000, 0.1, 90.0]]})
        s = status.collect("ds1", cfg=_cfg(trainer), runner=self._runner(probe))
        assert s["best"]["metric"] == "validation/mel_loss" and s["best"]["step"] == 1000

    def test_probe_runs_in_the_trainer_venv_over_stdin(self, tmp_path):
        trainer = _trainer_tree(tmp_path)
        calls = []
        status.collect("ds1", cfg=_cfg(trainer), audio_out=tmp_path / "wavs",
                       runner=self._runner(self._probe(), calls))
        (cmd, kwargs), = calls
        assert cmd[0].endswith("python.exe") and cmd[1] == "-"  # script over stdin
        assert cmd[2].endswith(str(Path("checkpoints") / "ds1"))
        assert "EventAccumulator" in kwargs["input"]
        assert kwargs["env"]["SIGNALML_AUDIO_OUT"] == str(tmp_path / "wavs")
        assert kwargs["env"]["PYTHONUTF8"] == "1"  # IPA-safe, like the trainer itself

    def test_probe_failure_is_reported_not_swallowed(self, tmp_path):
        trainer = _trainer_tree(tmp_path)

        class Failed:
            returncode = 1
            stdout = ""
            stderr = "ModuleNotFoundError: tensorboard"

        s = status.collect("ds1", cfg=_cfg(trainer), runner=lambda cmd, **k: Failed())
        assert "tensorboard" in s["error"]
        assert "tensorboard" in status.format_status(s)

    def test_no_event_files_yet_reads_as_a_note(self, tmp_path):
        trainer = _trainer_tree(tmp_path)
        probe = {"exp_dir": "/exp", "error": "no event files yet", "scalars": {},
                 "checkpoints": [], "audio": []}
        s = status.collect("ds1", cfg=_cfg(trainer), runner=self._runner(probe))
        assert s["step"] == 0 and "no event files" in status.format_status(s)
