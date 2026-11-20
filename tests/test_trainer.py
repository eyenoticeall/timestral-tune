"""Tests for the training loop: `head_only` end to end, evaluation, schedules, stopping, failure modes."""

from __future__ import annotations

import copy
import dataclasses
import math

import pytest
import torch

from timestral_tune import TimestralTrainer, TrainingConfig
from timestral_tune.config import DEFAULT_LEARNING_RATES
from timestral_tune.trainer import Callback, EarlyStopping, EvalResult, compute_loss
from timestral_tune.trainer import trainer as trainer_module
from timestral_tune.trainer.trainer import _learning_rate_factor

from .conftest import available_accelerators
from .helpers import make_toy_dataset

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

    def poisoned(model, batch, frame):
        output = real(model, batch, frame)
        return dataclasses.replace(output, loss=output.loss * float("nan"))

    monkeypatch.setattr(trainer_module, "compute_loss", poisoned)
    with pytest.raises(FloatingPointError, match="Lower the learning rate"):
        trainer.fit()
    assert not trainer.model.training, "the model is left in eval mode even when training fails"


def test_features_of_later_phases_raise_not_implemented(tiny_forecaster):
    dataset = make_toy_dataset(tiny_forecaster.patch_size)
    model = fresh_model(tiny_forecaster)
    for overrides in ({"mode": "lora_variate"}, {"mode": "full_ft"}, {"mixed_precision": "bf16"}):
        with pytest.raises(NotImplementedError, match="Phase 2"):
            TimestralTrainer(config_for(**overrides), dataset, model=model)
    with pytest.raises(NotImplementedError, match="Phase 2"):
        TimestralTrainer(config_for(), dataset, lora_config=object(), model=model)
    with pytest.raises(NotImplementedError, match="Phase 2"):
        TimestralTrainer(config_for(), dataset, model=model).export()


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
    ],
)
def test_config_rejects_invalid_values(overrides):
    with pytest.raises(ValueError):
        config_for(**overrides)


def test_config_defaults_follow_the_spec():
    config = TrainingConfig(model_name_or_path="mistralai/Timestral-28M")
    assert config.mode == "head_only" and config.target_frame == "median_folded" and config.mixed_precision == "fp32"
    assert config.gradient_clip_val == 1.0 and config.lr_schedule == "cosine"
    assert config.resolved_learning_rate == 1e-4
    assert dataclasses.replace(config, mode="lora_variate").resolved_learning_rate == 3e-4
    assert dataclasses.replace(config, mode="full_ft", learning_rate=5e-6).resolved_learning_rate == 5e-6
    assert TrainingConfig(model_name_or_path="x", gradient_clip_val=None).gradient_clip_val is None
