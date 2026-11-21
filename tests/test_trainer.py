"""Tests for the training loop: `head_only` end to end, evaluation, schedules, stopping, failure modes."""

from __future__ import annotations

import copy
import dataclasses
import math
import shutil

import numpy as np
import pytest
import torch

from timestral_tune import LoraConfig, TimestralTrainer, TrainingConfig
from timestral_tune.config import DEFAULT_LEARNING_RATES
from timestral_tune.models import merged_state_dict
from timestral_tune.trainer import Callback, EarlyStopping, EvalResult, compute_loss
from timestral_tune.trainer import trainer as trainer_module
from timestral_tune.trainer.trainer import _learning_rate_factor

from .conftest import available_accelerators
from .helpers import l2_rel, make_toy_dataset

DEVICES = ["cpu", *available_accelerators()]


def fresh_model(forecaster):
    """A private copy: training mutates the model, and the fixtures are shared across tests."""
    return copy.deepcopy(forecaster.model)


def config_for(device: str = "cpu", **overrides) -> TrainingConfig:
    settings = {"model_name_or_path": "unused: a model is passed in", "device": device, "log_every": 1000}
    return TrainingConfig(**{**settings, **overrides})


def fixed_batch_config(dataset, device: str, **overrides) -> TrainingConfig:
    """Every step sees the same batch: the whole dataset, unshuffled."""
    return config_for(device, batch_size=len(dataset), shuffle=False, **overrides)


def snapshot(model) -> dict[str, torch.Tensor]:
    return {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()}


# ---------------------------------------------------------- end to end


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("covariates", [False, True], ids=["univariate", "covariates"])
def test_head_only_overfits_a_fixed_batch_and_touches_only_the_head(
    forecaster, device, covariates, is_released_checkpoint
):
    """Roadmap Phase 1: `head_only` training end to end on a toy dataset, on CPU and the Apple GPU.

    Overfitting one fixed batch must lower its loss, and every parameter outside the head must stay
    bit-identical.
    """
    model = fresh_model(forecaster)
    dataset = make_toy_dataset(forecaster.patch_size, covariates=covariates)
    learning_rate = 1e-3 if is_released_checkpoint else 1e-2
    config = fixed_batch_config(dataset, device, max_steps=25, learning_rate=learning_rate, lr_schedule="constant")
    before = snapshot(model)

    trainer = TimestralTrainer(config, dataset, model=model)
    loss_before = trainer.evaluate(dataset).loss
    result = trainer.fit()
    loss_after = trainer.evaluate(dataset).loss

    assert result.steps == 25 and not result.stopped_early and result.skipped_batches == 0
    assert all(math.isfinite(loss) for _, loss in result.train_history)
    assert loss_after < loss_before, f"loss did not fall: {loss_before:.5f} -> {loss_after:.5f}"
    assert result.train_history[0][1] == pytest.approx(loss_before, rel=1e-4), "step 1 starts from the pretrained loss"

    after = snapshot(model)
    head = {name for name in after if name.startswith("head.")}
    assert head and len(head) < len(after)
    assert all(torch.equal(after[name], before[name]) for name in after if name not in head), "a frozen weight moved"
    assert all(not torch.equal(after[name], before[name]) for name in head), "a head weight did not train"
    assert {name for name, p in model.named_parameters() if p.requires_grad} == head
    assert trainer.num_trainable_parameters == sum(after[name].numel() for name in head)
    assert next(model.parameters()).device.type == device and not model.training


def test_training_is_deterministic_for_a_seed(tiny_forecaster):
    dataset = make_toy_dataset(tiny_forecaster.patch_size)
    heads = []
    for _ in range(2):
        model = fresh_model(tiny_forecaster)
        TimestralTrainer(config_for(batch_size=4, max_steps=12, learning_rate=1e-2, seed=7), dataset, model=model).fit()
        heads.append(torch.cat([p.detach().flatten() for p in model.head.parameters()]))
    assert torch.equal(heads[0], heads[1])


def test_the_trainer_loads_the_checkpoint_it_is_given(tiny_forecaster, tmp_path):
    """Without `model=`, the trainer loads `model_name_or_path` through the pinned, device-aware loader."""
    from safetensors.torch import save_file

    tiny_forecaster.model.config.save(tmp_path / "config.json")
    save_file(tiny_forecaster.model.state_dict(), str(tmp_path / "model.safetensors"))
    dataset = make_toy_dataset(tiny_forecaster.patch_size)
    trainer = TimestralTrainer(TrainingConfig(model_name_or_path=tmp_path, device="cpu", max_steps=2), dataset)
    assert trainer.fit().steps == 2
    assert trainer.config.resolved_learning_rate == DEFAULT_LEARNING_RATES["head_only"]


# ------------------------------------------------------------ evaluation


def test_evaluate_is_the_mean_over_scored_positions_not_over_batches(tiny_forecaster):
    model = fresh_model(tiny_forecaster)
    dataset = make_toy_dataset(tiny_forecaster.patch_size, series=5)
    # Make the windows contribute unequal numbers of scored positions.
    dataset.series[0].targets[:, -3:] = float("nan")
    trainer = TimestralTrainer(config_for(batch_size=4, eval_batch_size=3), dataset, model=model)
    result = trainer.evaluate(dataset)

    collate = dataset.collator(model.config)
    whole = collate([dataset[i] for i in range(len(dataset))])
    with torch.no_grad():
        expected = compute_loss(model, whole).loss.item()
    assert result.loss == pytest.approx(expected, rel=1e-5)
    assert result.scored_positions == whole.num_scored
    assert result.batches == math.ceil(len(dataset) / 3)
    assert not model.training


def test_evaluate_needs_a_dataset(tiny_forecaster):
    trainer = TimestralTrainer(
        config_for(), make_toy_dataset(tiny_forecaster.patch_size), model=fresh_model(tiny_forecaster)
    )
    with pytest.raises(ValueError, match="no dataset to evaluate"):
        trainer.evaluate()


# ------------------------------------------- validation, stopping, restore


class Recorder(Callback):
    def __init__(self, stop_at: int | None = None) -> None:
        self.steps: list[int] = []
        self.rates: list[float] = []
        self.evaluations: list[int] = []
        self.started = self.ended = 0
        self.stop_at = stop_at

    def on_fit_start(self, trainer):
        self.started += 1

    def on_step_end(self, trainer, step, loss, learning_rate):
        self.steps.append(step)
        self.rates.append(learning_rate)
        if step == self.stop_at:
            trainer.should_stop = True

    def on_evaluate(self, trainer, step, result):
        assert isinstance(result, EvalResult)
        self.evaluations.append(step)

    def on_fit_end(self, trainer, result):
        self.ended += 1


def test_evaluation_cadence_and_callbacks(tiny_forecaster):
    dataset = make_toy_dataset(tiny_forecaster.patch_size)
    train, validation = dataset.split_by_time(12 * tiny_forecaster.patch_size)
    recorder = Recorder()
    config = config_for(batch_size=4, max_steps=7, eval_every=3, learning_rate=1e-2, warmup_steps=2)
    result = TimestralTrainer(config, train, validation, callbacks=[recorder], model=fresh_model(tiny_forecaster)).fit()
    assert recorder.steps == list(range(1, 8))
    assert recorder.evaluations == [3, 6, 7], "every 3 steps, plus once at the end"
    assert [step for step, _ in result.eval_history] == [3, 6, 7]
    assert recorder.started == recorder.ended == 1
    assert recorder.rates[0] == pytest.approx(1e-2 / 3) and recorder.rates[2] == pytest.approx(1e-2)
    assert result.best_step in {3, 6, 7} and result.best_eval_loss == min(loss for _, loss in result.eval_history)


def test_epoch_end_evaluation_when_eval_every_is_unset(tiny_forecaster):
    dataset = make_toy_dataset(tiny_forecaster.patch_size)
    train, validation = dataset.split_by_time(12 * tiny_forecaster.patch_size)
    steps_per_epoch = math.ceil(len(train) / 4)
    result = TimestralTrainer(
        config_for(batch_size=4, epochs=3), train, validation, model=fresh_model(tiny_forecaster)
    ).fit()
    assert result.steps == 3 * steps_per_epoch and result.epochs == 3
    assert [step for step, _ in result.eval_history] == [steps_per_epoch, 2 * steps_per_epoch, 3 * steps_per_epoch]


def test_a_callback_can_stop_training(tiny_forecaster):
    dataset = make_toy_dataset(tiny_forecaster.patch_size)
    trainer = TimestralTrainer(
        config_for(batch_size=4, max_steps=50),
        dataset,
        callbacks=[Recorder(stop_at=3)],
        model=fresh_model(tiny_forecaster),
    )
    result = trainer.fit()
    assert result.steps == 3 and result.stopped_early and trainer.global_step == 3


def test_a_callback_can_stop_training_before_the_first_step(tiny_forecaster):
    class StopAtOnce(Recorder):
        def on_fit_start(self, trainer):
            super().on_fit_start(trainer)
            trainer.should_stop = True

    dataset = make_toy_dataset(tiny_forecaster.patch_size)
    callback = StopAtOnce()
    model = fresh_model(tiny_forecaster)
    before = snapshot(model)
    result = TimestralTrainer(config_for(batch_size=4, max_steps=5), dataset, callbacks=[callback], model=model).fit()
    assert result.steps == 0 and result.stopped_early and result.final_train_loss is None
    assert result.train_history == [] and callback.steps == [] and callback.ended == 1
    assert all(torch.equal(value, before[name]) for name, value in snapshot(model).items())


def test_callbacks_must_be_callback_instances(tiny_forecaster):
    class OnlyOneHook:
        def on_step_end(self, trainer, step, loss, learning_rate): ...

    dataset = make_toy_dataset(tiny_forecaster.patch_size)
    with pytest.raises(TypeError, match="Callback instances, got OnlyOneHook"):
        TimestralTrainer(config_for(), dataset, callbacks=[OnlyOneHook()], model=fresh_model(tiny_forecaster))


def test_early_stopping_counts_evaluations_without_improvement():
    class Stub:
        should_stop = False

    stopper, stub = EarlyStopping(patience=2, min_delta=0.01), Stub()
    stopper.on_fit_start(stub)
    for step, loss in enumerate([1.0, 0.5, 0.495, 0.6], start=1):  # the third is within min_delta of the best
        stopper.on_evaluate(stub, step, EvalResult(loss=loss, scored_positions=1, batches=1))
        assert stub.should_stop == (step == 4)
    assert stopper.best == 0.5
    with pytest.raises(ValueError):
        EarlyStopping(patience=0)


def test_early_stopping_from_the_config_stops_a_run_that_only_gets_worse(tiny_forecaster):
    dataset = make_toy_dataset(tiny_forecaster.patch_size)
    train, validation = dataset.split_by_time(12 * tiny_forecaster.patch_size)
    # A learning rate this large makes the head worse at every step.
    config = config_for(batch_size=len(train), shuffle=False, max_steps=40, eval_every=1, learning_rate=0.5,
                        lr_schedule="constant", early_stopping_patience=3)  # fmt: skip
    model = fresh_model(tiny_forecaster)
    trainer = TimestralTrainer(config, train, validation, model=model)
    result = trainer.fit()
    assert result.stopped_early and result.steps < 40
    assert result.steps == result.best_step + 3
    assert result.restored_best
    assert trainer.evaluate().loss == pytest.approx(result.best_eval_loss, rel=1e-5), "the best head was not restored"


def test_restore_best_can_be_turned_off(tiny_forecaster):
    dataset = make_toy_dataset(tiny_forecaster.patch_size)
    train, validation = dataset.split_by_time(12 * tiny_forecaster.patch_size)
    config = config_for(batch_size=len(train), shuffle=False, max_steps=6, eval_every=1, learning_rate=0.5,
                        lr_schedule="constant", restore_best=False)  # fmt: skip
    trainer = TimestralTrainer(config, train, validation, model=fresh_model(tiny_forecaster))
    result = trainer.fit()
    assert not result.restored_best
    assert trainer.evaluate().loss == pytest.approx(result.eval_history[-1][1], rel=1e-5)


def test_fit_can_be_called_again_with_a_fresh_schedule(tiny_forecaster):
    dataset = make_toy_dataset(tiny_forecaster.patch_size)
    recorder = Recorder()
    trainer = TimestralTrainer(
        config_for(batch_size=4, max_steps=5, learning_rate=1e-2),
        dataset,
        callbacks=[recorder],
        model=fresh_model(tiny_forecaster),
    )
    first, second = trainer.fit(), trainer.fit()
    assert first.steps == second.steps == 5 and trainer.global_step == 10
    assert recorder.steps == list(range(1, 11))
    assert recorder.rates[5] == pytest.approx(1e-2), "the second run starts again at the peak learning rate"
    assert recorder.rates[:5] == pytest.approx(recorder.rates[5:])


def test_progress_is_logged_per_fit_call(tiny_forecaster, caplog):
    """The global step keeps counting across `fit` calls; the progress line counts within one call."""
    dataset = make_toy_dataset(tiny_forecaster.patch_size)
    trainer = TimestralTrainer(
        config_for(batch_size=4, max_steps=3, log_every=2), dataset, model=fresh_model(tiny_forecaster)
    )
    with caplog.at_level("INFO", logger="timestral_tune.trainer"):
        trainer.fit()
        trainer.fit()
    progress = [record.getMessage().split("  ")[0] for record in caplog.records if "/3  loss" in record.getMessage()]
    assert progress == ["step 2/3", "step 3/3", "step 2/3", "step 3/3"]
    assert trainer.global_step == 6 and trainer.fit_start_step == 3


# ------------------------------------------------- optimisation settings


def grad_norm(trainer) -> float:
    return torch.cat([p.grad.flatten() for p in trainer.model.head.parameters()]).norm().item()


class GradNorms(Callback):
    def __init__(self) -> None:
        self.norms: list[float] = []

    def on_step_end(self, trainer, step, loss, learning_rate):
        self.norms.append(grad_norm(trainer))


def test_gradients_are_clipped_to_the_configured_norm(tiny_forecaster):
    dataset = make_toy_dataset(tiny_forecaster.patch_size)
    norms = {}
    for clip in (None, 1e-3):
        recorder = GradNorms()
        config = fixed_batch_config(dataset, "cpu", max_steps=3, gradient_clip_val=clip)
        TimestralTrainer(config, dataset, callbacks=[recorder], model=fresh_model(tiny_forecaster)).fit()
        norms[clip] = recorder.norms
    assert min(norms[None]) > 1e-2, "unclipped gradients must exceed the clip value for the test to mean anything"
    assert norms[1e-3] == pytest.approx([1e-3] * 3, rel=1e-4)


def test_weight_decay_reaches_the_optimiser_and_shrinks_the_head(tiny_forecaster):
    dataset = make_toy_dataset(tiny_forecaster.patch_size)
    norms = {}
    for decay in (0.0, 0.5):
        model = fresh_model(tiny_forecaster)
        config = fixed_batch_config(dataset, "cpu", max_steps=8, learning_rate=1e-2, weight_decay=decay)
        trainer = TimestralTrainer(config, dataset, model=model)
        assert trainer.optimizer.param_groups[0]["weight_decay"] == decay
        trainer.fit()
        norms[decay] = torch.cat([p.detach().flatten() for p in model.head.parameters()]).norm().item()
    assert norms[0.5] < norms[0.0]


@pytest.mark.parametrize("schedule", ["cosine", "constant"])
def test_the_learning_rate_the_optimiser_uses_follows_the_schedule(tiny_forecaster, schedule):
    dataset = make_toy_dataset(tiny_forecaster.patch_size)
    recorder = Recorder()
    config = config_for(batch_size=4, max_steps=12, warmup_steps=2, learning_rate=1e-2, lr_schedule=schedule)
    TimestralTrainer(config, dataset, callbacks=[recorder], model=fresh_model(tiny_forecaster)).fit()
    rates = recorder.rates
    assert rates[:3] == pytest.approx([1e-2 / 3, 2e-2 / 3, 1e-2]), "linear warm-up to the peak"
    if schedule == "constant":
        assert rates[2:] == pytest.approx([1e-2] * 10)
    else:
        assert all(a > b for a, b in zip(rates[2:], rates[3:], strict=False)), "decays after the peak"
        assert 0 < rates[-1] < 1e-3, "close to zero, but still positive, on the last step"


def test_shuffle_changes_the_batch_order_and_the_seed_fixes_it(tiny_forecaster):
    dataset = make_toy_dataset(tiny_forecaster.patch_size, series=8)
    collate = dataset.collator(tiny_forecaster.model.config)
    in_order = [
        collate([dataset[i] for i in range(start, start + 4)]).targets for start in range(0, len(dataset) - 3, 4)
    ]

    def first_epoch(**overrides) -> list[torch.Tensor]:
        trainer = TimestralTrainer(config_for(batch_size=4, **overrides), dataset, model=fresh_model(tiny_forecaster))
        return [batch.targets for batch in trainer._loader(dataset, training=True)][: len(in_order)]

    def same(a, b) -> bool:
        return all(torch.equal(x, y) for x, y in zip(a, b, strict=True))

    assert same(first_epoch(shuffle=False), in_order)
    shuffled = first_epoch(shuffle=True, seed=1)
    assert not same(shuffled, in_order)
    assert same(shuffled, first_epoch(shuffle=True, seed=1)) and not same(shuffled, first_epoch(shuffle=True, seed=2))


def test_the_configured_target_frame_is_the_one_trained_and_evaluated_in(tiny_forecaster, monkeypatch):
    patch = tiny_forecaster.patch_size
    dataset = make_toy_dataset(patch)
    model = fresh_model(tiny_forecaster)
    whole = dataset.collator(model.config)([dataset[i] for i in range(len(dataset))])
    with torch.no_grad():
        expected = {frame: compute_loss(model, whole, frame).loss.item() for frame in ("median_folded", "true_values")}
    assert expected["median_folded"] != pytest.approx(expected["true_values"], rel=1e-3), "the frames must differ here"

    seen = []
    real = trainer_module.compute_loss
    monkeypatch.setattr(
        trainer_module,
        "compute_loss",
        lambda m, b, frame, **kwargs: seen.append(frame) or real(m, b, frame, **kwargs),
    )
    for frame in ("median_folded", "true_values"):
        config = fixed_batch_config(dataset, "cpu", max_steps=1, target_frame=frame)
        trainer = TimestralTrainer(config, dataset, model=fresh_model(tiny_forecaster))
        assert trainer.evaluate(dataset).loss == pytest.approx(expected[frame], rel=1e-5)
        assert trainer.fit().train_history[0][1] == pytest.approx(expected[frame], rel=1e-5)
    assert set(seen[:2]) == {"median_folded"} and set(seen[2:]) == {"true_values"}


# ---------------------------------------------------------- learning rate


def test_learning_rate_schedule():
    factors = [_learning_rate_factor(step, 3, 13, "cosine") for step in range(13)]
    assert factors[:3] == pytest.approx([0.25, 0.5, 0.75]) and factors[3] == 1.0
    assert all(a > b for a, b in zip(factors[3:], factors[4:], strict=False)), "decays monotonically after warm-up"
    assert 0 < factors[-1] < 0.1, "the last step still has a positive learning rate"
    assert _learning_rate_factor(13, 3, 13, "cosine") == pytest.approx(0.0, abs=1e-12)
    assert [_learning_rate_factor(step, 2, 10, "constant") for step in (0, 1, 2, 9)] == pytest.approx(
        [1 / 3, 2 / 3, 1, 1]
    )
    assert _learning_rate_factor(0, 0, 1, "cosine") == 1.0


# ---------------------------------------------------------- failure modes


def test_unscored_batches_are_skipped_and_counted(tiny_forecaster, monkeypatch):
    model = fresh_model(tiny_forecaster)
    dataset = make_toy_dataset(tiny_forecaster.patch_size)
    collate = dataset.collator(model.config)
    scored = collate([dataset[0], dataset[1]])
    unscored = dataclasses.replace(scored, loss_weight=torch.zeros_like(scored.loss_weight))
    trainer = TimestralTrainer(config_for(max_steps=4), dataset, model=model)
    monkeypatch.setattr(trainer, "_loader", lambda dataset, training: [scored, unscored, scored])
    result = trainer.fit()
    assert result.steps == 4 and result.skipped_batches == 2 and result.epochs == 2

    monkeypatch.setattr(trainer, "_loader", lambda dataset, training: [unscored, unscored])
    with pytest.raises(RuntimeError, match="no optimiser step"):
        trainer.fit()
    with pytest.raises(ValueError, match="no scored position"):
        trainer._evaluate([unscored])


def test_a_non_finite_loss_stops_training_with_an_explanation(tiny_forecaster, monkeypatch):
    dataset = make_toy_dataset(tiny_forecaster.patch_size)
    trainer = TimestralTrainer(config_for(max_steps=3), dataset, model=fresh_model(tiny_forecaster))
    real = trainer_module.compute_loss

    def poisoned(*args, **kwargs):
        output = real(*args, **kwargs)
        return dataclasses.replace(output, loss=output.loss * float("nan"))

    monkeypatch.setattr(trainer_module, "compute_loss", poisoned)
    with pytest.raises(FloatingPointError, match="Lower the learning rate"):
        trainer.fit()
    assert not trainer.model.training, "the model is left in eval mode even when training fails"


def test_trainer_mode_precision_and_compile_validation(tiny_forecaster):
    dataset = make_toy_dataset(tiny_forecaster.patch_size, covariates=True)
    uni = make_toy_dataset(tiny_forecaster.patch_size, covariates=False)

    # bf16 raises on non-CUDA devices (CPU and MPS).
    for device in [d for d in DEVICES if d != "cuda"]:
        with pytest.raises(ValueError, match="CUDA only"):
            TimestralTrainer(config_for(device, mixed_precision="bf16"), dataset, model=fresh_model(tiny_forecaster))

    # lora_config rejected for non-LoRA modes and wrong type.
    for mode in ("head_only", "full_ft"):
        with pytest.raises(ValueError, match="LoRA modes"):
            TimestralTrainer(
                config_for(mode=mode), dataset, lora_config=LoraConfig(), model=fresh_model(tiny_forecaster)
            )
    with pytest.raises(TypeError, match="LoraConfig"):
        TimestralTrainer(
            config_for(mode="lora_variate"), dataset, lora_config=object(), model=fresh_model(tiny_forecaster)
        )  # type: ignore[arg-type]

    # Mode vs LoraConfig axis mismatch rejected.
    with pytest.raises(ValueError, match="requires 'variate'"):
        TimestralTrainer(
            config_for(mode="lora_variate"),
            dataset,
            lora_config=LoraConfig(axes=("time",)),
            model=fresh_model(tiny_forecaster),
        )
    with pytest.raises(ValueError, match="requires 'time'"):
        TimestralTrainer(
            config_for(mode="lora_temporal"),
            dataset,
            lora_config=LoraConfig(axes=("variate",)),
            model=fresh_model(tiny_forecaster),
        )

    # Single-row guard enforced by TimestralTrainer (T10 end-to-end).
    with pytest.raises(ValueError, match="single-row"):
        TimestralTrainer(config_for(mode="lora_variate"), uni, model=fresh_model(tiny_forecaster))
    with pytest.raises(ValueError, match="inert"):
        TimestralTrainer(
            config_for(mode="lora_variate"),
            uni,
            lora_config=LoraConfig(axes=("variate",), targets=("q", "v"), allow_single_row=True),
            model=fresh_model(tiny_forecaster),
        )

    # compile_mode="reduce-overhead" rejects non-divisible dataset sizes.
    assert len(dataset) % 5 != 0
    with pytest.raises(ValueError, match="fixed batch shapes"):
        TimestralTrainer(
            config_for(batch_size=5, compile_model=True, compile_mode="reduce-overhead"),
            dataset,
            model=fresh_model(tiny_forecaster),
        )


def test_incompatible_datasets_and_models_are_rejected_at_construction(tiny_forecaster):
    import timestral_tune._vendor.timestral as vendored
    from timestral_tune.data import TimeSeriesDataset

    patch, model = tiny_forecaster.patch_size, fresh_model(tiny_forecaster)
    dataset = make_toy_dataset(patch)
    too_long = TimeSeriesDataset(dataset.series, context_length=8 * patch, forecast_horizon=5 * patch)
    with pytest.raises(ValueError, match="exceeds the 16 steps"):
        TimestralTrainer(config_for(), too_long, model=model)
    too_much_context = TimeSeriesDataset(dataset.series, context_length=17 * patch, forecast_horizon=patch)
    with pytest.raises(ValueError, match="context cap"):
        TimestralTrainer(config_for(), too_much_context, model=model)
    with pytest.raises(ValueError, match="same columns per role"):
        TimestralTrainer(config_for(), dataset, make_toy_dataset(patch, covariates=True), model=model)
    with pytest.raises(ValueError, match="needs a val_dataset"):
        TimestralTrainer(config_for(early_stopping_patience=2), dataset, model=model)
    with pytest.raises(TypeError, match="TuneTimestralModel"):
        TimestralTrainer(config_for(), dataset, model=vendored.TimestralModel(model.config))

    no_covariates = dataclasses.replace(model.config, covariate_enabled=False)
    with pytest.raises(ValueError, match="not trained with covariates"):
        TimestralTrainer(config_for(), make_toy_dataset(patch, covariates=True), model=type(model)(no_covariates))
    time_only = dataclasses.replace(model.config, block_schedule=("time", "time"))
    with pytest.raises(ValueError, match="cannot condition on"):
        TimestralTrainer(config_for(), make_toy_dataset(patch, covariates=True), model=type(model)(time_only))


# ---------------------------------------------------------------- config


@pytest.mark.parametrize(
    "overrides",
    [
        {"mode": "everything"},
        {"mixed_precision": "fp16"},
        {"lr_schedule": "linear"},
        {"target_frame": "oracle"},
        {"learning_rate": 0.0},
        {"learning_rate": float("nan")},
        {"weight_decay": -1.0},
        {"gradient_clip_val": 0.0},
        {"batch_size": 0},
        {"epochs": 0},
        {"max_steps": 0},
        {"eval_every": 0},
        {"early_stopping_patience": 0},
        {"warmup_steps": -1},
        {"num_workers": -1},
        {"batch_size": 2.5},
        {"seed": True},
        {"batch_size": np.bool_(True)},
        {"max_steps": np.float64(3.0)},
        {"cache_frozen_features": "yes"},
        {"compile_model": 1},
        {"compile_mode": "max-autotune"},
    ],
)
def test_config_rejects_invalid_values(overrides):
    with pytest.raises(ValueError):
        config_for(**overrides)


def test_config_accepts_numpy_integers_and_stores_python_ints(tiny_forecaster):
    """Values often come from an array or a sweep. They are validated, then stored as plain `int`."""
    config = config_for(batch_size=np.int64(4), max_steps=np.int32(2), seed=np.uint8(3), warmup_steps=np.int64(0))
    for name in ("batch_size", "max_steps", "seed", "warmup_steps"):
        assert type(getattr(config, name)) is int
    dataset = make_toy_dataset(tiny_forecaster.patch_size)
    assert TimestralTrainer(config, dataset, model=fresh_model(tiny_forecaster)).fit().steps == 2
    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        config_for(batch_size=np.int64(0))


def test_config_defaults_follow_the_spec():
    config = TrainingConfig(model_name_or_path="mistralai/Timestral-28M")
    assert config.mode == "head_only" and config.target_frame == "median_folded" and config.mixed_precision == "fp32"
    assert config.cache_frozen_features is True
    assert config.gradient_clip_val == 1.0 and config.lr_schedule == "cosine"
    assert config.resolved_learning_rate == 1e-4
    assert dataclasses.replace(config, mode="lora_variate").resolved_learning_rate == 3e-4
    assert dataclasses.replace(config, mode="full_ft", learning_rate=5e-6).resolved_learning_rate == 5e-6
    assert TrainingConfig(model_name_or_path="x", gradient_clip_val=None).gradient_clip_val is None


# -------------------------------------------------- frozen-feature cache


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("covariates", [False, True], ids=["univariate", "covariates"])
@pytest.mark.parametrize("frame", ["median_folded", "true_values"])
def test_head_only_cache_matches_uncached_training(tiny_forecaster, device, covariates, frame):
    dataset = make_toy_dataset(tiny_forecaster.patch_size, covariates=covariates)
    train, val = dataset.split_by_time(12 * tiny_forecaster.patch_size)
    results = {}
    heads = {}
    for cached in (False, True):
        model = fresh_model(tiny_forecaster)
        config = config_for(
            device,
            batch_size=4,
            epochs=4,
            learning_rate=1e-2,
            target_frame=frame,
            seed=5,
            cache_frozen_features=cached,
        )
        trainer = TimestralTrainer(config, train, val, model=model)
        results[cached] = trainer.fit()
        heads[cached] = torch.cat([p.detach().cpu().flatten() for p in model.head.parameters()])

    assert results[True].steps == results[False].steps
    assert results[True].best_eval_loss == pytest.approx(results[False].best_eval_loss, rel=1e-4)
    assert results[True].final_train_loss == pytest.approx(results[False].final_train_loss, rel=1e-4)
    # With shuffle=True, cached hidden states are built in unshuffled batches, so float32 GEMM tiling can
    # shift an individual coordinate by ~5e-5 over 4 AdamW steps on MPS (PyTorch 2.8); overall L2 stays < 1e-4.
    assert l2_rel(heads[True], heads[False]) < 1e-4
    torch.testing.assert_close(heads[True], heads[False], rtol=1e-2, atol=1e-4)


@pytest.mark.parametrize("device", DEVICES)
def test_head_only_cache_runs_transformer_once_per_dataset(tiny_forecaster, device):
    dataset = make_toy_dataset(tiny_forecaster.patch_size, covariates=True)
    train, val = dataset.split_by_time(12 * tiny_forecaster.patch_size)

    def count_body_calls(cached: bool) -> int:
        model = fresh_model(tiny_forecaster)
        calls = 0

        def bump(*_):
            nonlocal calls
            calls += 1

        handle = model.transformer.register_forward_hook(bump)
        try:
            config = config_for(
                device, batch_size=len(train), epochs=5, learning_rate=1e-2, cache_frozen_features=cached
            )
            trainer = TimestralTrainer(config, train, val, model=model)
            trainer.fit()
            trainer.evaluate()
        finally:
            handle.remove()
        return calls

    cached_calls = count_body_calls(True)
    uncached_calls = count_body_calls(False)
    # Cached: 1 pass to build train cache + 1 pass to build val cache.
    # Uncached: 5 train epochs + 5 end-of-epoch evals + 1 explicit evaluate() = 11 passes.
    assert cached_calls == 2, f"expected 2 body passes with cache, got {cached_calls}"
    assert uncached_calls == 11, f"expected 11 body passes without cache, got {uncached_calls}"


def test_head_only_cache_falls_back_when_over_memory_cap(tiny_forecaster, monkeypatch):
    dataset = make_toy_dataset(tiny_forecaster.patch_size, covariates=True)
    model = fresh_model(tiny_forecaster)
    calls = 0

    def bump(*_):
        nonlocal calls
        calls += 1

    monkeypatch.setattr(trainer_module, "_MAX_HEAD_CACHE_BYTES", 1)
    handle = model.transformer.register_forward_hook(bump)
    try:
        config = fixed_batch_config(dataset, "cpu", max_steps=4, learning_rate=1e-2, cache_frozen_features=True)
        trainer = TimestralTrainer(config, dataset, model=model)
        assert trainer.fit().steps == 4
    finally:
        handle.remove()
    assert calls == 4, "expected streaming fallback to run the body on every step"


# ------------------------------------ LoRA, full_ft, bf16 and torch.compile


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("mode", ["lora_variate", "lora_temporal", "full_ft"])
def test_lora_and_full_ft_overfit_and_touch_only_selected_parameters(tiny_forecaster, device, mode):
    dataset = make_toy_dataset(tiny_forecaster.patch_size, covariates=True)
    model = fresh_model(tiny_forecaster)
    lr = 5e-3 if mode == "full_ft" else 2e-2
    config = fixed_batch_config(dataset, device, mode=mode, max_steps=15, learning_rate=lr, lr_schedule="constant")
    lora_cfg = (
        None
        if mode == "full_ft"
        else LoraConfig(r=4, lora_alpha=8, axes=("variate" if mode == "lora_variate" else "time",))
    )
    trainer = TimestralTrainer(config, dataset, lora_config=lora_cfg, model=model)

    before = {name: p.detach().cpu().clone() for name, p in model.named_parameters()}
    loss_before = trainer.evaluate(dataset).loss
    result = trainer.fit()
    loss_after = trainer.evaluate(dataset).loss

    assert result.steps == 15 and loss_after < 0.7 * loss_before, (
        f"{mode} on {device}: {loss_before:.4f} -> {loss_after:.4f}"
    )
    changed = {name for name, p in model.named_parameters() if not torch.equal(p.detach().cpu(), before[name])}
    if mode == "full_ft":
        assert any(name.startswith("head.") for name in changed)
        assert any(name.startswith("transformer.") for name in changed)
    else:
        assert changed and all(".lora_" in name for name in changed), f"non-LoRA weights changed: {changed}"
        assert trainer.num_trainable_parameters == sum(p.numel() for n, p in model.named_parameters() if ".lora_" in n)


def test_lora_variate_single_row_override_trains_cleanly(tiny_forecaster):
    uni = make_toy_dataset(tiny_forecaster.patch_size, covariates=False)
    model = fresh_model(tiny_forecaster)
    cfg = fixed_batch_config(uni, "cpu", mode="lora_variate", max_steps=10, learning_rate=2e-2, lr_schedule="constant")
    lora_cfg = LoraConfig(r=4, lora_alpha=8, axes=("variate",), targets=("v", "o"), allow_single_row=True)
    trainer = TimestralTrainer(cfg, uni, lora_config=lora_cfg, model=model)
    loss_before = trainer.evaluate(uni).loss
    trainer.fit()
    assert trainer.evaluate(uni).loss < loss_before


@pytest.mark.parametrize("mode", ["lora_variate", "lora_temporal", "full_ft"])
def test_lora_and_full_ft_are_deterministic_for_a_seed(tiny_forecaster, mode):
    dataset = make_toy_dataset(tiny_forecaster.patch_size, covariates=True)
    weights = []
    for _ in range(2):
        model = fresh_model(tiny_forecaster)
        config = config_for("cpu", mode=mode, batch_size=4, max_steps=6, learning_rate=1e-3, seed=11)
        trainer = TimestralTrainer(config, dataset, model=model)
        trainer.fit()
        weights.append(merged_state_dict(model))
    for k in weights[0]:
        assert torch.equal(weights[0][k], weights[1][k]), f"{mode} diverged on {k}"


@pytest.mark.skipif(not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(), reason="requires bf16 CUDA")
@pytest.mark.parametrize("mode", ["head_only", "lora_variate", "lora_temporal", "full_ft"])
def test_bf16_training_on_cuda(tiny_forecaster, mode):
    dataset = make_toy_dataset(tiny_forecaster.patch_size, covariates=True)
    model = fresh_model(tiny_forecaster)
    lr = 5e-3 if mode == "full_ft" else 2e-2
    config = fixed_batch_config(
        dataset, "cuda", mode=mode, mixed_precision="bf16", max_steps=10, learning_rate=lr, lr_schedule="constant"
    )
    trainer = TimestralTrainer(config, dataset, model=model)
    loss_before = trainer.evaluate(dataset).loss
    result = trainer.fit()
    loss_after = trainer.evaluate(dataset).loss
    assert result.steps == 10 and math.isfinite(loss_after) and loss_after < loss_before


def test_compile_model_trains_and_produces_clean_merged_state_dict(tiny_forecaster):
    from timestral_tune._vendor.timestral import TimestralModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and shutil.which("g++") is None and shutil.which("clang++") is None:
        pytest.skip("CPU torch.compile requires a C++ compiler (g++ or clang++)")
    dataset = make_toy_dataset(tiny_forecaster.patch_size, covariates=True)
    model = fresh_model(tiny_forecaster)
    config = fixed_batch_config(
        dataset,
        device,
        mode="lora_variate",
        max_steps=4,
        learning_rate=2e-2,
        lr_schedule="constant",
        compile_model=True,
        compile_mode="default",
    )
    trainer = TimestralTrainer(config, dataset, model=model)
    loss_before = trainer.evaluate(dataset).loss
    result = trainer.fit()
    loss_after = trainer.evaluate(dataset).loss
    assert result.steps == 4 and loss_after < loss_before
    state = merged_state_dict(model)
    assert all("_orig_mod" not in k for k in state)
    reloaded = TimestralModel(model.config)
    reloaded.load_state_dict(state, strict=True)


def test_lora_trainer_init_does_not_mutate_global_rng_and_evaluate_validates_inputs(tiny_forecaster):
    dataset = make_toy_dataset(tiny_forecaster.patch_size, covariates=True)
    torch.manual_seed(1234)
    rng_before = torch.get_rng_state()
    trainer = TimestralTrainer(
        config_for("cpu", mode="lora_variate", batch_size=len(dataset), max_steps=1, seed=99),
        dataset,
        model=fresh_model(tiny_forecaster),
    )
    assert torch.equal(torch.get_rng_state(), rng_before), "trainer __init__ mutated global PyTorch RNG"

    with pytest.raises(TypeError, match="TimeSeriesDataset"):
        trainer.evaluate("not-a-dataset")  # type: ignore[arg-type]

    # evaluate(other_dataset) enforces compile_mode="reduce-overhead" fixed batch shapes too.
    compiled_trainer = TimestralTrainer(
        config_for(
            "cpu",
            mode="head_only",
            batch_size=len(dataset),
            eval_batch_size=5,
            max_steps=1,
            compile_model=True,
            compile_mode="reduce-overhead",
            cache_frozen_features=False,
        ),
        dataset,
        model=fresh_model(tiny_forecaster),
    )
    assert len(dataset) % 5 != 0
    with pytest.raises(ValueError, match="fixed batch shapes"):
        compiled_trainer.evaluate(dataset)


def test_trainer_resumes_from_load_lora_and_rejects_mismatched_pre_injected_model(tiny_forecaster, tmp_path):
    from timestral_tune import load_lora

    dataset = make_toy_dataset(tiny_forecaster.patch_size, covariates=True)
    cfg = fixed_batch_config(
        dataset, "cpu", mode="lora_variate", max_steps=4, learning_rate=2e-2, lr_schedule="constant"
    )
    lora_cfg = LoraConfig(r=4, lora_alpha=8, axes=("variate",), targets=("q", "v"))

    trainer1 = TimestralTrainer(cfg, dataset, lora_config=lora_cfg, model=fresh_model(tiny_forecaster))
    trainer1.fit()
    loss_after_1 = trainer1.evaluate(dataset).loss
    out_dir = trainer1.export(tmp_path / "resume-exp")

    # Reload adapters onto a fresh base model and resume training in a new TimestralTrainer.
    resumed_model, reloaded_cfg = load_lora(fresh_model(tiny_forecaster), out_dir)
    trainer2 = TimestralTrainer(cfg, dataset, lora_config=reloaded_cfg, model=resumed_model)
    assert trainer2.evaluate(dataset).loss == pytest.approx(loss_after_1, rel=1e-6)
    trainer2.fit()
    assert trainer2.evaluate(dataset).loss < loss_after_1

    # Passing a pre-injected LoRA model with a conflicting config or into head_only raises cleanly.
    with pytest.raises(ValueError, match="does not match"):
        TimestralTrainer(cfg, dataset, lora_config=LoraConfig(r=2, axes=("variate",)), model=resumed_model)
    with pytest.raises(ValueError, match="already has LoRA"):
        TimestralTrainer(
            fixed_batch_config(dataset, "cpu", mode="head_only", max_steps=1), dataset, model=resumed_model
        )
