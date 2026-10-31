from __future__ import annotations

__version__ = "1.0.0"

from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor


class BatchClosure:
    """Evaluate a deterministic loss on an already-selected fixed batch.

    ``loss_fn`` must close over the batch and return a scalar loss.  This
    lightweight helper does not replay random-number-generator or module-buffer
    state.  Use :class:`ReplayableBatchClosure` for modules such as Dropout or
    BatchNorm.
    """

    def __init__(self, loss_fn: Callable[[], Tensor]) -> None:
        if not callable(loss_fn):
            raise TypeError("BatchClosure requires a callable loss function")
        self._loss_fn = loss_fn

    def value_and_grad(self) -> Tensor:
        """Evaluate the loss and populate parameter gradients."""

        loss = self._loss_fn()
        loss.backward()
        return loss

    def value(self) -> Tensor:
        """Evaluate the loss without calling ``backward``."""

        return self._loss_fn()


@dataclass
class _ForwardState:
    cpu_rng: Tensor
    cuda_rng: list[Tensor] | None
    buffers: list[tuple[Tensor, Tensor]]


class ReplayableBatchClosure(BatchClosure):
    """Replay stochastic and stateful forwards under identical conditions.

    The first value-and-gradient evaluation records the CPU/CUDA random state
    and every registered buffer in ``model``.  Each value-only search trial is
    evaluated after restoring that same pre-forward state.  Once the optimizer
    finishes the replay interval, the state left by one ordinary base-point
    forward is restored, whether a trial was accepted, rejected, or raised.

    Randomness outside PyTorch and mutable state that is not registered as a
    model buffer are outside this contract.
    """

    def __init__(
        self, model: torch.nn.Module, loss_fn: Callable[[], Tensor]
    ) -> None:
        if not isinstance(model, torch.nn.Module):
            raise TypeError("ReplayableBatchClosure requires a torch.nn.Module")
        super().__init__(loss_fn)
        self._model = model
        self._before_forward: _ForwardState | None = None
        self._after_base_forward: _ForwardState | None = None

    def _capture_forward_state(self) -> _ForwardState:
        cuda_rng = None
        if torch.cuda.is_available():
            cuda_rng = [state.clone() for state in torch.cuda.get_rng_state_all()]
        return _ForwardState(
            cpu_rng=torch.get_rng_state().clone(),
            cuda_rng=cuda_rng,
            buffers=[
                (buffer, buffer.detach().clone()) for buffer in self._model.buffers()
            ],
        )

    @staticmethod
    def _restore_forward_state(snapshot: _ForwardState) -> None:
        torch.set_rng_state(snapshot.cpu_rng)
        if snapshot.cuda_rng is not None:
            torch.cuda.set_rng_state_all(snapshot.cuda_rng)
        with torch.no_grad():
            for buffer, value in snapshot.buffers:
                if (
                    buffer.shape != value.shape
                    or buffer.dtype != value.dtype
                    or buffer.device != value.device
                ):
                    raise RuntimeError(
                        "a model buffer changed shape, dtype, or device during replay"
                    )
                buffer.copy_(value)

    def _yamadam_value_and_grad_no_replay(self) -> Tensor:
        """Evaluate one base forward without opening a replay interval.

        YamAdam calls this private fast path only for the exact built-in
        closure type after the optimizer has completed first-step
        calibration.  A pre-forward snapshot is retained locally for
        exception rollback.  On success, the live RNG and model buffers
        already represent one ordinary base forward, so no post-forward
        snapshot or restore is needed.
        """

        if (
            self._before_forward is not None
            or self._after_base_forward is not None
        ):
            raise RuntimeError("ReplayableBatchClosure is already active")
        before_forward: _ForwardState | None = None
        forward_started = False
        try:
            before_forward = self._capture_forward_state()
            forward_started = True
            loss = BatchClosure.value_and_grad(self)
        except BaseException:
            try:
                if before_forward is not None and forward_started:
                    self._restore_forward_state(before_forward)
            finally:
                # Capture, forward, backward, and rollback failures must not
                # publish or retain a partial public replay interval.
                self._before_forward = None
                self._after_base_forward = None
            raise
        self._before_forward = None
        self._after_base_forward = None
        return loss

    def value_and_grad(self) -> Tensor:
        if self._before_forward is not None:
            raise RuntimeError("ReplayableBatchClosure is already active")
        self._before_forward = self._capture_forward_state()
        try:
            loss = super().value_and_grad()
            self._after_base_forward = self._capture_forward_state()
            return loss
        except BaseException:
            try:
                self._restore_forward_state(self._before_forward)
            finally:
                self._before_forward = None
                self._after_base_forward = None
            raise

    def value(self) -> Tensor:
        if self._before_forward is None:
            return super().value()
        self._restore_forward_state(self._before_forward)
        return super().value()

    def _restore_base_forward(self) -> None:
        """Restore the state produced by the ordinary base-point forward."""

        try:
            if self._after_base_forward is not None:
                self._restore_forward_state(self._after_base_forward)
        finally:
            self._before_forward = None
            self._after_base_forward = None

import math
from typing import ClassVar


@dataclass(frozen=True)
class ScaleProposal:
    """A validated radius and the group metadata to commit with it."""

    radius: float
    updates: dict[str, float | str]


class NormalizedInnovationScale:
    """Self-normalized recursion retained as a falsified reference."""

    name: ClassVar[str] = "normalized_innovation"

    @classmethod
    def ensure_group(cls, group: dict) -> None:
        group.setdefault("yamadam_scale_name", cls.name)
        group.setdefault("yamadam_running_square", 0.0)
        group.setdefault("yamadam_accumulator", 0.0)
        group.setdefault("yamadam_last_z", 0.0)

    @classmethod
    def initialize(
        cls, *, calibration_radius: float, s: float, omega: float
    ) -> ScaleProposal:
        del omega
        square = s * s
        if (
            not math.isfinite(calibration_radius)
            or calibration_radius <= 0.0
            or not math.isfinite(square)
            or square <= 0.0
        ):
            raise FloatingPointError("normalized scale initialization is invalid")
        return ScaleProposal(
            radius=calibration_radius,
            updates={
                "yamadam_scale_name": cls.name,
                "yamadam_running_square": square,
                "yamadam_accumulator": 1.0,
                "yamadam_last_z": 1.0,
            },
        )

    @classmethod
    def propose(cls, group: dict, *, s: float, omega: float) -> ScaleProposal:
        square = s * s
        running_square = (
            (1.0 - omega) * float(group["yamadam_running_square"])
            + omega * square
        )
        normalization_square = max(running_square, square)
        if normalization_square <= 0.0 or not math.isfinite(normalization_square):
            raise FloatingPointError("normalized innovation RMS is invalid")
        normalized = s / math.sqrt(normalization_square)
        accumulator = float(group["yamadam_accumulator"]) + (
            omega * normalized * normalized
        )
        safety_square = max(accumulator, normalized * normalized)
        radius = (
            float(group["yamadam_radius"])
            * normalized
            / math.sqrt(safety_square)
        )
        if (
            not math.isfinite(running_square)
            or not math.isfinite(normalized)
            or normalized <= 0.0
            or not math.isfinite(accumulator)
            or not math.isfinite(safety_square)
            or safety_square <= 0.0
            or not math.isfinite(radius)
            or radius <= 0.0
        ):
            raise FloatingPointError("normalized innovation scale is invalid")
        return ScaleProposal(
            radius=radius,
            updates={
                "yamadam_scale_name": cls.name,
                "yamadam_running_square": running_square,
                "yamadam_accumulator": accumulator,
                "yamadam_last_z": normalized,
            },
        )

    @classmethod
    def diagnostics(cls, group: dict) -> dict[str, float | str]:
        return {
            "scale_rule": cls.name,
            "R": float(group["yamadam_running_square"]),
            "A": float(group["yamadam_accumulator"]),
            "z": float(group["yamadam_last_z"]),
        }


class _RawCumulativeScale:
    """Shared implementation of raw cumulative directional energy."""

    name: ClassVar[str]

    @classmethod
    def ensure_group(cls, group: dict) -> None:
        group.setdefault("yamadam_scale_name", cls.name)
        group.setdefault("yamadam_energy", 0.0)
        group.setdefault("yamadam_last_h", 0.0)
        group.setdefault("yamadam_last_energy_weight", 0.0)

    @classmethod
    def _weight(cls, omega: float) -> float:
        raise NotImplementedError

    @classmethod
    def initialize(
        cls, *, calibration_radius: float, s: float, omega: float
    ) -> ScaleProposal:
        del omega
        square = s * s
        if (
            not math.isfinite(calibration_radius)
            or calibration_radius <= 0.0
            or not math.isfinite(square)
            or square <= 0.0
        ):
            raise FloatingPointError("raw scale initialization is invalid")
        # The accepted Armijo displacement already supplies the first update;
        # its directional square is intentionally not charged to later steps.
        return ScaleProposal(
            radius=calibration_radius,
            updates={
                "yamadam_scale_name": cls.name,
                "yamadam_energy": 0.0,
                "yamadam_last_h": square,
                "yamadam_last_energy_weight": 0.0,
            },
        )

    @classmethod
    def propose(cls, group: dict, *, s: float, omega: float) -> ScaleProposal:
        square = s * s
        weight = cls._weight(omega)
        energy = float(group["yamadam_energy"]) + weight * square
        h = max(energy, square)
        radius = float(group["yamadam_radius"]) * s / math.sqrt(h)
        if (
            not math.isfinite(weight)
            or weight <= 0.0
            or not math.isfinite(energy)
            or energy < 0.0
            or not math.isfinite(h)
            or h <= 0.0
            or not math.isfinite(radius)
            or radius <= 0.0
        ):
            raise FloatingPointError("raw cumulative scale is invalid")
        return ScaleProposal(
            radius=radius,
            updates={
                "yamadam_scale_name": cls.name,
                "yamadam_energy": energy,
                "yamadam_last_h": h,
                "yamadam_last_energy_weight": weight,
            },
        )

    @classmethod
    def diagnostics(cls, group: dict) -> dict[str, float | str]:
        return {
            "scale_rule": cls.name,
            "E": float(group["yamadam_energy"]),
            "H": float(group["yamadam_last_h"]),
            "kappa": float(group["yamadam_last_energy_weight"]),
        }


class RawCumulativeScale(_RawCumulativeScale):
    """Unweighted ``E += s^2`` recursion."""

    name = "raw_cumulative"

    @classmethod
    def _weight(cls, omega: float) -> float:
        del omega
        return 1.0


class InnovationWeightedRawCumulativeScale(_RawCumulativeScale):
    """Innovation-weighted ``E += omega*s^2`` recursion."""

    name = "innovation_weighted_raw_cumulative"

    @classmethod
    def _weight(cls, omega: float) -> float:
        return omega


class _CalibratedInverseSqrtScale:
    """Use the first accepted radius on a square-root update clock.

    The first nonzero update has index one and radius ``C``.  Every later
    nonzero update increments the internal clock and uses ``C/sqrt(k)``.
    Zero directions do not advance this clock because no displacement is
    applied.  The rule has no workload-facing setting.
    """

    name: ClassVar[str] = "calibrated_inverse_sqrt"

    @classmethod
    def ensure_group(cls, group: dict) -> None:
        group.setdefault("yamadam_scale_name", cls.name)
        group.setdefault("yamadam_scale_update", 0.0)

    @classmethod
    def initialize(
        cls, *, calibration_radius: float, s: float, omega: float
    ) -> ScaleProposal:
        del omega
        if (
            not math.isfinite(calibration_radius)
            or calibration_radius <= 0.0
            or not math.isfinite(s)
            or s <= 0.0
        ):
            raise FloatingPointError("inverse-square-root scale initialization is invalid")
        return ScaleProposal(
            radius=calibration_radius,
            updates={
                "yamadam_scale_name": cls.name,
                "yamadam_scale_update": 1.0,
            },
        )

    @classmethod
    def propose(cls, group: dict, *, s: float, omega: float) -> ScaleProposal:
        del omega
        previous_update = float(group["yamadam_scale_update"])
        update = previous_update + 1.0
        calibration_radius = float(group["yamadam_radius"])
        radius = calibration_radius / math.sqrt(update)
        if (
            not math.isfinite(s)
            or s <= 0.0
            or not math.isfinite(previous_update)
            or previous_update < 1.0
            or not math.isfinite(update)
            or not math.isfinite(calibration_radius)
            or calibration_radius <= 0.0
            or not math.isfinite(radius)
            or radius <= 0.0
        ):
            raise FloatingPointError("inverse-square-root scale state is invalid")
        return ScaleProposal(
            radius=radius,
            updates={
                "yamadam_scale_name": cls.name,
                "yamadam_scale_update": update,
            },
        )

    @classmethod
    def diagnostics(cls, group: dict) -> dict[str, float | str]:
        return {
            "scale_rule": cls.name,
            "k": float(group["yamadam_scale_update"]),
        }


class _GuardedInverseSqrtScale:
    """Use the smaller of a clock radius and a relative-gradient radius.

    Let ``C`` be the accepted first radius, ``s_k`` the positive directional
    derivative on update ``k``, and

    ``Q_k = sum_{i<=k} s_i^2``
    ``z_k = s_k / sqrt(Q_k / k)``
    ``A_k = sum_{i<=k} z_i^2``.

    The applied radius is

    ``min(C/sqrt(k), C*z_k/sqrt(A_k))``.

    The first term supplies a deterministic upper envelope.  The second makes
    the radius vanish with a directional derivative that becomes small
    relative to its own history.  Both terms are invariant to one fixed
    positive rescaling of every loss in the run.
    """

    name: ClassVar[str] = "guarded_inverse_sqrt"

    @classmethod
    def ensure_group(cls, group: dict) -> None:
        group.setdefault("yamadam_scale_name", cls.name)
        group.setdefault("yamadam_scale_update", 0.0)
        group.setdefault("yamadam_scale_square_sum", 0.0)
        group.setdefault("yamadam_scale_accumulator", 0.0)
        group.setdefault("yamadam_scale_last_z", 0.0)
        group.setdefault("yamadam_scale_clock_radius", 0.0)
        group.setdefault("yamadam_scale_relative_radius", 0.0)

    @classmethod
    def initialize(
        cls, *, calibration_radius: float, s: float, omega: float
    ) -> ScaleProposal:
        del omega
        square = s * s
        if (
            not math.isfinite(calibration_radius)
            or calibration_radius <= 0.0
            or not math.isfinite(square)
            or square <= 0.0
        ):
            raise FloatingPointError("guarded scale initialization is invalid")
        return ScaleProposal(
            radius=calibration_radius,
            updates={
                "yamadam_scale_name": cls.name,
                "yamadam_scale_update": 1.0,
                "yamadam_scale_square_sum": square,
                "yamadam_scale_accumulator": 1.0,
                "yamadam_scale_last_z": 1.0,
                "yamadam_scale_clock_radius": calibration_radius,
                "yamadam_scale_relative_radius": calibration_radius,
            },
        )

    @classmethod
    def propose(cls, group: dict, *, s: float, omega: float) -> ScaleProposal:
        del omega
        previous_update = float(group["yamadam_scale_update"])
        update = previous_update + 1.0
        square = s * s
        square_sum = float(group["yamadam_scale_square_sum"]) + square
        mean_square = square_sum / update
        normalized = s / math.sqrt(mean_square)
        accumulator = float(group["yamadam_scale_accumulator"]) + (
            normalized * normalized
        )
        calibration_radius = float(group["yamadam_radius"])
        clock_radius = calibration_radius / math.sqrt(update)
        relative_radius = (
            calibration_radius * normalized / math.sqrt(accumulator)
        )
        radius = min(clock_radius, relative_radius)
        values = (
            previous_update,
            update,
            square,
            square_sum,
            mean_square,
            normalized,
            accumulator,
            calibration_radius,
            clock_radius,
            relative_radius,
            radius,
        )
        if (
            not all(math.isfinite(value) for value in values)
            or previous_update < 1.0
            or update <= previous_update
            or any(value <= 0.0 for value in values[2:])
        ):
            raise FloatingPointError("guarded scale state is invalid")
        return ScaleProposal(
            radius=radius,
            updates={
                "yamadam_scale_name": cls.name,
                "yamadam_scale_update": update,
                "yamadam_scale_square_sum": square_sum,
                "yamadam_scale_accumulator": accumulator,
                "yamadam_scale_last_z": normalized,
                "yamadam_scale_clock_radius": clock_radius,
                "yamadam_scale_relative_radius": relative_radius,
            },
        )

    @classmethod
    def diagnostics(cls, group: dict) -> dict[str, float | str]:
        return {
            "scale_rule": cls.name,
            "k": float(group["yamadam_scale_update"]),
            "Q": float(group["yamadam_scale_square_sum"]),
            "A": float(group["yamadam_scale_accumulator"]),
            "z": float(group["yamadam_scale_last_z"]),
            "clock_radius": float(group["yamadam_scale_clock_radius"]),
            "relative_radius": float(group["yamadam_scale_relative_radius"]),
        }


class _CumulativeRMSScaleBase:
    """Shared implementation of cumulative RMS-normalized radii.

    These policies are deliberately private while they are benchmark
    candidates.  Both retain the first directional square, unlike the raw
    cumulative policies above, so a vanishing later innovation also produces
    a vanishing radius.
    """

    name: ClassVar[str]

    @classmethod
    def ensure_group(cls, group: dict) -> None:
        group.setdefault("yamadam_scale_name", cls.name)
        group.setdefault("yamadam_rms_square_sum", 0.0)
        group.setdefault("yamadam_rms_weight_sum", 0.0)
        group.setdefault("yamadam_rms_accumulator", 0.0)
        group.setdefault("yamadam_rms_last_z", 0.0)
        group.setdefault("yamadam_rms_time", 0.0)

    @classmethod
    def _weight(cls, time: float) -> float:
        raise NotImplementedError

    @classmethod
    def initialize(
        cls, *, calibration_radius: float, s: float, omega: float
    ) -> ScaleProposal:
        del omega
        square = s * s
        if (
            not math.isfinite(calibration_radius)
            or calibration_radius <= 0.0
            or not math.isfinite(square)
            or square <= 0.0
        ):
            raise FloatingPointError("cumulative RMS initialization is invalid")
        return ScaleProposal(
            radius=calibration_radius,
            updates={
                "yamadam_scale_name": cls.name,
                "yamadam_rms_square_sum": square,
                "yamadam_rms_weight_sum": 1.0,
                "yamadam_rms_accumulator": 1.0,
                "yamadam_rms_last_z": 1.0,
                "yamadam_rms_time": 1.0,
            },
        )

    @classmethod
    def propose(cls, group: dict, *, s: float, omega: float) -> ScaleProposal:
        del omega
        square = s * s
        previous_time = float(group["yamadam_rms_time"])
        time = previous_time + 1.0
        weight = cls._weight(time)
        square_sum = float(group["yamadam_rms_square_sum"]) + weight * square
        weight_sum = float(group["yamadam_rms_weight_sum"]) + weight
        mean_square = square_sum / weight_sum
        if (
            not math.isfinite(square)
            or square <= 0.0
            or not math.isfinite(previous_time)
            or previous_time < 1.0
            or not math.isfinite(time)
            or not math.isfinite(weight)
            or weight <= 0.0
            or not math.isfinite(square_sum)
            or square_sum <= 0.0
            or not math.isfinite(weight_sum)
            or weight_sum <= 0.0
            or not math.isfinite(mean_square)
            or mean_square <= 0.0
        ):
            raise FloatingPointError("cumulative RMS state is invalid")
        normalized = s / math.sqrt(mean_square)
        accumulator = float(group["yamadam_rms_accumulator"]) + (
            normalized * normalized
        )
        radius = (
            float(group["yamadam_radius"])
            * normalized
            / math.sqrt(accumulator)
        )
        if (
            not math.isfinite(normalized)
            or normalized <= 0.0
            or not math.isfinite(accumulator)
            or accumulator <= 0.0
            or not math.isfinite(radius)
            or radius <= 0.0
            or radius > float(group["yamadam_radius"])
        ):
            raise FloatingPointError("cumulative RMS radius is invalid")
        return ScaleProposal(
            radius=radius,
            updates={
                "yamadam_scale_name": cls.name,
                "yamadam_rms_square_sum": square_sum,
                "yamadam_rms_weight_sum": weight_sum,
                "yamadam_rms_accumulator": accumulator,
                "yamadam_rms_last_z": normalized,
                "yamadam_rms_time": time,
            },
        )

    @classmethod
    def diagnostics(cls, group: dict) -> dict[str, float | str]:
        return {
            "scale_rule": cls.name,
            "Q": float(group["yamadam_rms_square_sum"]),
            "W": float(group["yamadam_rms_weight_sum"]),
            "A": float(group["yamadam_rms_accumulator"]),
            "z": float(group["yamadam_rms_last_z"]),
            "t": float(group["yamadam_rms_time"]),
        }


class _CumulativeRMSScale(_CumulativeRMSScaleBase):
    """Uniform cumulative RMS: ``Q += s^2`` and ``W += 1``."""

    name = "cumulative_rms"

    @classmethod
    def _weight(cls, time: float) -> float:
        del time
        return 1.0


class _LinearWeightedCumulativeRMSScale(_CumulativeRMSScaleBase):
    """Linearly weighted RMS: ``Q += t*s^2`` and ``W += t``."""

    name = "linear_weighted_cumulative_rms"

    @classmethod
    def _weight(cls, time: float) -> float:
        return time


from collections.abc import Mapping, Sequence
from dataclasses import InitVar, asdict
from typing import Any, Iterable

from torch.optim import Optimizer



@dataclass(frozen=True)
class YamAdamStepInfo:
    """Diagnostics for the most recently committed optimizer step."""

    accepted: bool
    reason: str
    loss_before: float
    loss_after: float
    radius: float
    directional_derivative: float
    omega: float
    scale: dict[str, float | str]
    projection_old_dot: float = 0.0
    projection_coefficient: float = 0.0
    projection_margin: float = 0.0
    projection_active: bool = False
    trials: int = 0
    expansions: int = 0
    backtracks: int = 0
    # Compatibility-only inputs used by private benchmark subclasses.
    # InitVar fields are absent from asdict() and from public diagnostics.
    local_resets: InitVar[int] = 0
    conflicts: InitVar[int] = 0


@dataclass
class _Candidate:
    parameter: Tensor
    gradient: Tensor
    mean: Tensor
    second_moment: Tensor
    gamma: Tensor
    direction: Tensor
    old_direction: Tensor
    current_direction: Tensor
    count: int
    history_state: dict | None = None


@dataclass
class _Direction:
    candidates: list[_Candidate]
    loss: float
    direction_norm: float
    slope: float
    directional_gradient: float
    omega: float
    coordinate_count: int
    gamma_min: float
    gamma_max: float
    projection_old_dot: float
    projection_gradient_square: float
    projection_coefficient: float
    projection_active: bool
    projection_margin: float = 0.0

    # Compatibility shims for private benchmark step implementations.  The
    # production direction performs no coordinate-local reset or hard gate.
    @property
    def local_resets(self) -> int:
        return 0

    @property
    def conflicts(self) -> int:
        return 0


class _YamAdamCore(Optimizer):
    """Parameter-only optimizer with automatic memory and step scale.

    For each coordinate, ``gamma`` is the reciprocal effective history.  If
    ``u_old`` is the previous mean/RMS direction, the memory update is

    ``a = 2*g*m_old / (g^2 + max(v_old, m_old^2))``
    ``c = (1 + a) / 2``
    ``beta = c / (gamma + c)``
    ``gamma_new = gamma / (gamma + c)``.

    The same ``beta`` is used for the first and second moments.  The complete
    preconditioned old-memory vector is projected continuously onto the global
    gradient half-space, then the current innovation is added.  The resulting
    direction is normalized globally before applying the scalar radius rule
    described in :meth:`step`.
    """

    _MAX_INITIAL_TRIALS = 64
    _ARMIJO_RATIO = 0.5
    _SCALE_POLICY = _GuardedInverseSqrtScale
    def __init__(self, params: Iterable[Tensor]):
        parameters = list(params)
        if not parameters:
            raise ValueError("YamAdam received an empty parameter list")
        if any(isinstance(item, dict) for item in parameters):
            raise TypeError(
                "YamAdam has one global scale and does not accept "
                "parameter-group dictionaries"
            )
        self._constructed = False
        super().__init__(parameters, defaults={})
        self._constructed = True
        self._ensure_group_state()
        self._last_info: YamAdamStepInfo | None = None
        # The common dense FP32/FP64 path stores the three coordinate-history
        # tensors in persistent flat buffers.  Per-parameter optimizer state
        # contains views of those buffers, while this private cache records
        # their ownership and permits the next step to reuse them without
        # concatenating the history again.  Neither object is serialized.
        self._flat_history_cache: dict[str, object] | None = None
        self._pending_flat_history: tuple[
            int,
            Tensor,
            Tensor,
            Tensor,
            tuple[tuple[Tensor, Tensor, Tensor], ...],
        ] | None = None

    def add_param_group(self, param_group: dict) -> None:
        if getattr(self, "_constructed", False):
            raise RuntimeError("YamAdam does not support adding parameter groups")
        super().add_param_group(param_group)

    def _ensure_group_state(self) -> None:
        group = self.param_groups[0]
        group.setdefault("yamadam_initialized", False)
        group.setdefault("yamadam_steps", 0)
        group.setdefault("yamadam_radius", 0.0)
        group.setdefault("yamadam_last_rho", 0.0)
        group.setdefault("yamadam_last_s", 0.0)
        group.setdefault("yamadam_last_omega", 1.0)
        group.setdefault("yamadam_coordinate_count", 0)
        group.setdefault("yamadam_gamma_min", 1.0)
        group.setdefault("yamadam_gamma_max", 1.0)
        group.setdefault("yamadam_gamma_bytes", 0)
        group.setdefault("yamadam_projection_old_dot", 0.0)
        group.setdefault("yamadam_projection_gradient_square", 0.0)
        group.setdefault("yamadam_projection_coefficient", 0.0)
        group.setdefault("yamadam_projection_margin", 0.0)
        group.setdefault("yamadam_projection_active", False)
        group.setdefault("yamadam_initial_trials", 0)
        group.setdefault("yamadam_initial_expansions", 0)
        group.setdefault("yamadam_initial_backtracks", 0)
        group.setdefault("yamadam_initial_curvature", 0.0)
        group.setdefault("yamadam_initial_expansion_blocked", False)
        group.setdefault("yamadam_initial_probe_radius", 0.0)
        group.setdefault("yamadam_initial_probe_trials", 0)
        group.setdefault("yamadam_initial_probe_expansions", 0)
        group.setdefault("yamadam_initial_probe_backtracks", 0)
        group.setdefault("yamadam_initial_probe_directional_gradient", 0.0)
        group.setdefault("yamadam_initial_probe_cosine", 0.0)
        group.setdefault("yamadam_initial_recheck_trials", 0)
        group.setdefault("yamadam_initial_recheck_backtracks", 0)
        group.setdefault("yamadam_initial_radius_limiter", "uninitialized")
        self._SCALE_POLICY.ensure_group(group)

    @property
    def last_info(self) -> dict[str, bool | float | int | str] | None:
        return None if self._last_info is None else asdict(self._last_info)

    @property
    def diagnostics(self) -> dict[str, bool | float | int | str]:
        """Return the committed global scale and memory diagnostics."""

        group = self.param_groups[0]
        result: dict[str, bool | float | int | str] = {
            "initialized": bool(group["yamadam_initialized"]),
            "steps": int(group["yamadam_steps"]),
            "C": float(group["yamadam_radius"]),
            "rho": float(group["yamadam_last_rho"]),
            "s": float(group["yamadam_last_s"]),
            "omega": float(group["yamadam_last_omega"]),
            "coordinate_count": int(group["yamadam_coordinate_count"]),
            "coordinate_gamma_min": float(group["yamadam_gamma_min"]),
            "coordinate_gamma_max": float(group["yamadam_gamma_max"]),
            "coordinate_state_bytes": int(group["yamadam_gamma_bytes"]),
            "projection_old_dot": float(group["yamadam_projection_old_dot"]),
            "projection_gradient_square": float(
                group["yamadam_projection_gradient_square"]
            ),
            "projection_coefficient": float(
                group["yamadam_projection_coefficient"]
            ),
            "projection_margin": float(group["yamadam_projection_margin"]),
            "projection_active": bool(group["yamadam_projection_active"]),
            "initial_trials": int(group["yamadam_initial_trials"]),
            "initial_expansions": int(group["yamadam_initial_expansions"]),
            "initial_backtracks": int(group["yamadam_initial_backtracks"]),
            "initial_curvature": float(group["yamadam_initial_curvature"]),
            "initial_expansion_blocked": bool(
                group["yamadam_initial_expansion_blocked"]
            ),
            "initial_probe_radius": float(
                group["yamadam_initial_probe_radius"]
            ),
            "initial_probe_trials": int(
                group["yamadam_initial_probe_trials"]
            ),
            "initial_probe_expansions": int(
                group["yamadam_initial_probe_expansions"]
            ),
            "initial_probe_backtracks": int(
                group["yamadam_initial_probe_backtracks"]
            ),
            "initial_probe_directional_gradient": float(
                group["yamadam_initial_probe_directional_gradient"]
            ),
            "initial_probe_cosine": float(
                group["yamadam_initial_probe_cosine"]
            ),
            "initial_recheck_trials": int(
                group["yamadam_initial_recheck_trials"]
            ),
            "initial_recheck_backtracks": int(
                group["yamadam_initial_recheck_backtracks"]
            ),
            "initial_radius_limiter": str(
                group["yamadam_initial_radius_limiter"]
            ),
        }
        result.update(self._SCALE_POLICY.diagnostics(group))
        return result

    @staticmethod
    def _state_dtype(parameter: Tensor) -> torch.dtype:
        if parameter.dtype in (torch.float16, torch.bfloat16):
            return torch.float32
        return parameter.dtype

    @staticmethod
    def _reduction_dtype(tensor: Tensor) -> torch.dtype:
        if tensor.dtype == torch.float64 or tensor.device.type == "cpu":
            return torch.float64
        return torch.float32

    @staticmethod
    def _safe_direction(mean: Tensor, second_moment: Tensor) -> Tensor:
        guarded_second = torch.maximum(second_moment.clamp_min(0), mean.square())
        denominator = guarded_second.sqrt()
        return torch.where(
            denominator > 0,
            mean / denominator,
            torch.zeros_like(mean),
        )

    @staticmethod
    def _validate_loss_tensor(loss: Tensor, *, where: str) -> None:
        if not isinstance(loss, Tensor):
            raise TypeError(f"closure must return a torch.Tensor ({where})")
        if loss.numel() != 1:
            raise ValueError(f"closure must return a scalar loss ({where})")

    @staticmethod
    def _loss_value(
        loss: Tensor, *, where: str, require_finite: bool = True
    ) -> float:
        YamAdam._validate_loss_tensor(loss, where=where)
        value = float(loss.detach())
        if require_finite and not math.isfinite(value):
            raise FloatingPointError(f"closure returned a non-finite loss ({where})")
        return value

    @staticmethod
    def _history_is_usable(
        parameter: Tensor, state: dict, previous_step: int
    ) -> bool:
        dtype = YamAdam._state_dtype(parameter)
        if int(state.get("last_seen_step", -1)) != previous_step:
            return False
        if int(state.get("count", 0)) <= 0:
            return False
        for name in ("mean", "second_moment", "gamma"):
            value = state.get(name)
            if (
                not isinstance(value, Tensor)
                or value.shape != parameter.shape
                or value.device != parameter.device
                or value.dtype != dtype
            ):
                return False
        return True

    @classmethod
    def _coordinate_consistency(
        cls,
        gradient: Tensor,
        old_mean: Tensor,
        old_second_moment: Tensor,
        *,
        gradient_square: Tensor | None = None,
    ) -> Tensor:
        """Return the coordinatewise fraction of old history to retain.

        Keeping this evidence rule separate from the moment recurrence makes
        alternative bounded agreement scores a local implementation change.
        """

        guarded_second = old_second_moment.clamp_min(0.0)
        torch.maximum(guarded_second, old_mean.square(), out=guarded_second)
        if gradient_square is None:
            gradient_square = gradient.square()
        torch.add(gradient_square, guarded_second, out=guarded_second)
        agreement = gradient.mul(2.0)
        agreement.mul_(old_mean).div_(guarded_second)
        agreement = torch.where(
            guarded_second > 0.0,
            agreement,
            torch.zeros_like(gradient),
        ).clamp_(-1.0, 1.0)
        torch.add(1.0, agreement, out=agreement)
        torch.mul(0.5, agreement, out=agreement)
        return agreement

    @staticmethod
    def _preconditioned_components(
        gradient: Tensor,
        old_mean: Tensor,
        beta: Tensor,
        gamma: Tensor,
        candidate_mean: Tensor,
        candidate_second: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Split the candidate direction into old and current components."""

        guarded_second = candidate_second.clamp_min(0.0)
        torch.maximum(guarded_second, candidate_mean.square(), out=guarded_second)
        denominator = guarded_second.sqrt_()
        nonzero = denominator > 0.0
        zero = torch.zeros_like(denominator)
        old_component = beta * old_mean
        old_component.div_(denominator)
        torch.where(nonzero, old_component, zero, out=old_component)
        current_component = gamma * gradient
        current_component.div_(denominator)
        torch.where(nonzero, current_component, zero, out=current_component)
        return old_component, current_component

    def _raw_gradients(self) -> list[tuple[Tensor, Tensor, dict]]:
        prepared: list[tuple[Tensor, Tensor, dict]] = []
        for parameter in self.param_groups[0]["params"]:
            gradient = parameter.grad
            if gradient is None:
                continue
            if gradient.is_sparse:
                raise RuntimeError("YamAdam does not support sparse gradients")
            if parameter.is_complex() or gradient.is_complex():
                raise RuntimeError("YamAdam does not support complex tensors")
            if not parameter.is_floating_point() or not gradient.is_floating_point():
                raise RuntimeError("YamAdam requires floating-point tensors")
            prepared.append(
                (
                    parameter,
                    gradient.detach().to(dtype=self._state_dtype(parameter)),
                    self.state.get(parameter, {}),
                )
            )
        return prepared

    @staticmethod
    def _flat_views(buffer: Tensor, references: list[Tensor]) -> list[Tensor]:
        """Return parameter-shaped views that exactly partition ``buffer``."""

        result: list[Tensor] = []
        offset = 0
        for reference in references:
            count = reference.numel()
            result.append(buffer.narrow(0, offset, count).view_as(reference))
            offset += count
        if offset != buffer.numel():
            raise RuntimeError("flat-buffer partition is inconsistent")
        return result

    def _cached_flat_history(
        self, states: list[dict], *, previous_step: int
    ) -> tuple[Tensor, Tensor, Tensor] | None:
        """Return owned flat history only when every state still uses it."""

        cache = self._flat_history_cache
        if cache is None or int(cache["step"]) != previous_step:
            return None
        views = cache["views"]
        if not isinstance(views, tuple) or len(views) != len(states):
            return None
        for state, item in zip(states, views):
            if (
                not isinstance(item, tuple)
                or len(item) != 3
                or state.get("mean") is not item[0]
                or state.get("second_moment") is not item[1]
                or state.get("gamma") is not item[2]
            ):
                return None
        buffers = cache["buffers"]
        if not isinstance(buffers, tuple) or len(buffers) != 3:
            return None
        return buffers  # type: ignore[return-value]

    def _detach_flat_history(self) -> None:
        """Release flat storage before entering a partial-state fallback.

        A tiny parameter view can otherwise keep an entire old full-model
        buffer alive after a missing-gradient step replaces only the active
        states.  Clone all still-attached views first, then publish the
        replacements together so an allocation failure leaves the old cache
        and state untouched.
        """

        self._pending_flat_history = None
        cache = self._flat_history_cache
        if cache is None:
            return
        views = cache.get("views")
        if not isinstance(views, tuple):
            self._flat_history_cache = None
            return

        replacements: list[
            tuple[dict, Tensor, Tensor, Tensor]
        ] = []
        for parameter, item in zip(self.param_groups[0]["params"], views):
            if not isinstance(item, tuple) or len(item) != 3:
                continue
            state = self.state.get(parameter)
            if not isinstance(state, dict):
                continue
            if (
                state.get("mean") is item[0]
                and state.get("second_moment") is item[1]
                and state.get("gamma") is item[2]
            ):
                replacements.append(
                    (
                        state,
                        item[0].detach().clone(),
                        item[1].detach().clone(),
                        item[2].detach().clone(),
                    )
                )
        for state, mean, second, gamma in replacements:
            state["mean"] = mean
            state["second_moment"] = second
            state["gamma"] = gamma
        self._flat_history_cache = None

    def _flat_layout(
        self,
        raw: list[tuple[Tensor, Tensor, dict]],
        *,
        previous_step: int,
    ) -> tuple[torch.device, torch.dtype, bool] | None:
        """Recognize only the homogeneous dense layout used by the fast path."""

        parameters = self.param_groups[0]["params"]
        if (
            not raw
            or len(parameters) < 2
            or len(raw) != len(parameters)
            or any(parameter is not row[0] for parameter, row in zip(parameters, raw))
            or any(parameter.numel() == 0 for parameter in parameters)
        ):
            return None

        device = raw[0][1].device
        dtype = raw[0][1].dtype
        if dtype not in (torch.float32, torch.float64):
            return None
        if any(
            parameter.device != device
            or gradient.device != device
            or parameter.dtype != dtype
            or gradient.dtype != dtype
            or self._state_dtype(parameter) != dtype
            or not parameter.is_contiguous()
            or not gradient.is_contiguous()
            for parameter, gradient, _state in raw
        ):
            return None

        usable = [
            self._history_is_usable(parameter, state, previous_step)
            for parameter, _gradient, state in raw
        ]
        if any(usable) and not all(usable):
            return None
        return device, dtype, all(usable)

    def _make_direction(
        self,
        raw: list[tuple[Tensor, Tensor, dict]],
        base_loss_tensor: Tensor,
        *,
        previous_step: int,
    ) -> _Direction:
        """Build a direction, fusing the usual homogeneous dense layout."""

        # A pending buffer belongs to one direction construction only.  This
        # also makes direct private-method use unable to retain an abandoned
        # candidate across a later call.
        self._pending_flat_history = None
        layout = self._flat_layout(raw, previous_step=previous_step)
        if layout is None:
            return self._make_direction_general(
                raw, base_loss_tensor, previous_step=previous_step
            )
        return self._make_flat_direction(
            raw,
            base_loss_tensor,
            previous_step=previous_step,
            layout=layout,
        )

    def _make_flat_direction(
        self,
        raw: list[tuple[Tensor, Tensor, dict]],
        base_loss_tensor: Tensor,
        *,
        previous_step: int,
        layout: tuple[torch.device, torch.dtype, bool],
    ) -> _Direction:
        """Fused form of the unchanged coordinate and projection equations."""

        _device, dtype, has_history = layout
        parameters = [row[0] for row in raw]
        reference_gradients = [row[1] for row in raw]
        states = [row[2] for row in raw]
        gradient = torch.cat([item.reshape(-1) for item in reference_gradients])
        gradient_square_tensor = gradient.square()
        reduction_dtype = self._reduction_dtype(gradient)
        direction_arithmetic_epsilon = torch.finfo(dtype).eps

        if has_history:
            cached = self._cached_flat_history(
                states, previous_step=previous_step
            )
            if cached is None:
                old_mean = torch.cat(
                    [state["mean"].reshape(-1) for state in states]
                )
                old_second = torch.cat(
                    [state["second_moment"].reshape(-1) for state in states]
                )
                old_gamma = torch.cat(
                    [state["gamma"].reshape(-1) for state in states]
                )
            else:
                old_mean, old_second, old_gamma = cached
            consistency = self._coordinate_consistency(
                gradient,
                old_mean,
                old_second,
                gradient_square=gradient_square_tensor,
            )
            memory_denominator = old_gamma + consistency
            torch.div(consistency, memory_denominator, out=consistency)
            beta = consistency
            torch.div(old_gamma, memory_denominator, out=memory_denominator)
            gamma = memory_denominator
            old_numerator = old_mean * beta
            mean = old_numerator + gradient * gamma
            second = old_second * beta
            second.add_(gradient_square_tensor * gamma)
            second.clamp_min_(0.0)
            torch.maximum(second, mean.square(), out=second)
            denominator = second.sqrt()
            nonzero = denominator > 0.0
            zero = torch.zeros_like(denominator)
            old_direction = old_numerator
            old_direction.div_(denominator)
            torch.where(nonzero, old_direction, zero, out=old_direction)
            direction = mean.div(denominator)
            torch.where(nonzero, direction, zero, out=direction)
            counts = [int(state["count"]) + 1 for state in states]
        else:
            old_mean = torch.empty(0, device=gradient.device, dtype=dtype)
            beta = torch.empty_like(old_mean)
            mean = gradient.clone()
            second = gradient_square_tensor
            gamma = torch.ones_like(gradient)
            old_direction = torch.zeros_like(gradient)
            denominator = gradient_square_tensor.sqrt()
            direction = torch.where(
                denominator > 0.0,
                gradient / denominator,
                torch.zeros_like(gradient),
            )
            counts = [1] * len(parameters)

        old_contribution = gradient * old_direction
        old_dot_tensor = torch.sum(
            old_contribution, dtype=reduction_dtype
        ).detach()
        old_absolute_dot_tensor = torch.sum(
            old_contribution.abs(), dtype=reduction_dtype
        ).detach()
        gradient_square_total_tensor = torch.sum(
            gradient_square_tensor, dtype=reduction_dtype
        ).detach()
        self._validate_loss_tensor(base_loss_tensor, where="base evaluation")
        loss_shares_transfer = (
            base_loss_tensor.device == gradient.device
            and base_loss_tensor.is_floating_point()
            and torch.finfo(base_loss_tensor.dtype).bits
            <= torch.finfo(reduction_dtype).bits
        )
        if loss_shares_transfer:
            loss_tensor = (
                base_loss_tensor.detach().to(reduction_dtype).reshape(())
            )
            loss = 0.0
        else:
            # Preserve the original scalar conversion for unusual closures
            # whose loss is on another device or cannot be represented
            # exactly in the metric transfer dtype.
            loss_tensor = torch.zeros_like(old_dot_tensor)
            loss = self._loss_value(base_loss_tensor, where="base evaluation")

        # The eligible flat layout has one device and one dtype.  A foreach
        # infinity norm preserves each tensor's exact max-absolute bound while
        # avoiding one Python-dispatched reduction launch per parameter.
        parameter_bounds = torch.stack(
            torch._foreach_norm(
                [parameter.detach() for parameter in parameters],
                float("inf"),
            )
        )
        gamma_min_tensor, gamma_max_tensor = torch.aminmax(gamma)
        second_min, second_max = torch.aminmax(second)
        pre_projection_bounds = torch.cat(
            (
                torch.stack(
                    (
                        gamma_min_tensor,
                        gamma_max_tensor,
                        second_min,
                        second_max,
                    )
                ),
                parameter_bounds,
            )
        )
        pre_projection_valid_tensor = torch.isfinite(
            pre_projection_bounds
        ).all()
        native_diagnostics = torch.stack(
            (gamma_min_tensor, gamma_max_tensor, second_min)
        )
        if native_diagnostics.dtype != reduction_dtype:
            native_diagnostics = native_diagnostics.to(reduction_dtype)
        first_values = (
            torch.cat(
                (
                    torch.stack(
                        (
                            old_dot_tensor,
                            old_absolute_dot_tensor,
                            gradient_square_total_tensor,
                            loss_tensor,
                        )
                    ),
                    native_diagnostics,
                    pre_projection_valid_tensor.to(reduction_dtype).reshape(1),
                )
            )
            .detach()
            .cpu()
            .tolist()
        )
        old_dot_total = float(first_values[0])
        old_absolute_dot_total = float(first_values[1])
        gradient_square_total = float(first_values[2])
        if loss_shares_transfer:
            loss = float(first_values[3])
            if not math.isfinite(loss):
                raise FloatingPointError(
                    "closure returned a non-finite loss (base evaluation)"
                )
        gamma_minimum = float(first_values[4])
        gamma_maximum = float(first_values[5])
        second_minimum = float(first_values[6])
        pre_projection_valid = bool(first_values[7])
        if (
            not pre_projection_valid
            or not math.isfinite(loss)
            or not math.isfinite(old_dot_total)
            or not math.isfinite(old_absolute_dot_total)
            or not math.isfinite(gradient_square_total)
            or old_absolute_dot_total < 0.0
            or gradient_square_total < 0.0
            or gamma_minimum <= 0.0
            or gamma_maximum > 1.0
            or second_minimum < 0.0
        ):
            raise FloatingPointError(
                "YamAdam encountered a non-finite loss, parameter, gradient, "
                "or moment state"
            )

        projection_margin = 64.0 * direction_arithmetic_epsilon * max(
            abs(old_dot_total), old_absolute_dot_total
        )
        projection_coefficient = (
            min(old_dot_total - projection_margin, 0.0)
            / gradient_square_total
            if gradient_square_total > 0.0
            else 0.0
        )
        if (
            not math.isfinite(projection_margin)
            or projection_margin < 0.0
            or not math.isfinite(projection_coefficient)
        ):
            raise FloatingPointError("YamAdam global projection is invalid")

        if projection_coefficient < 0.0:
            if has_history:
                # Preserve the active-projection evaluation order: combine
                # the already-computed old and current pieces only after the
                # shared half-space coefficient is known.
                direction.copy_(gamma).mul_(gradient).div_(denominator)
                torch.where(
                    nonzero,
                    direction,
                    torch.zeros_like(direction),
                    out=direction,
                )
                projected_old = old_direction
                projected_old.add_(
                    gradient, alpha=-projection_coefficient
                )
                direction.add_(projected_old)
            else:
                direction.add_(gradient, alpha=-projection_coefficient)

        contribution = gradient * direction
        slope_total_tensor = torch.sum(
            contribution, dtype=reduction_dtype
        ).detach()
        absolute_slope_total_tensor = torch.sum(
            contribution.abs(), dtype=reduction_dtype
        ).detach()
        direction_square_total_tensor = torch.sum(
            direction.square(), dtype=reduction_dtype
        ).detach()
        second_values = (
            torch.stack(
                (
                    slope_total_tensor,
                    absolute_slope_total_tensor,
                    direction_square_total_tensor,
                )
            )
            .detach()
            .cpu()
            .tolist()
        )
        slope_total = float(second_values[0])
        absolute_slope_total = float(second_values[1])
        direction_square_total = float(second_values[2])
        if (
            not math.isfinite(slope_total)
            or not math.isfinite(absolute_slope_total)
            or not math.isfinite(direction_square_total)
            or direction_square_total < 0.0
        ):
            raise FloatingPointError("YamAdam global direction is non-finite")
        slope_tolerance = 64.0 * direction_arithmetic_epsilon * max(
            abs(old_dot_total),
            abs(projection_coefficient) * gradient_square_total,
            absolute_slope_total,
        )
        if slope_total < -slope_tolerance:
            raise FloatingPointError(
                "YamAdam update direction has negative gradient alignment"
            )
        slope_total = max(0.0, slope_total)
        if slope_total == 0.0:
            directional_gradient = 0.0
        else:
            if direction_square_total <= 0.0:
                raise FloatingPointError("YamAdam produced a zero update direction")
            directional_gradient = slope_total / math.sqrt(direction_square_total)
        if not math.isfinite(directional_gradient):
            raise FloatingPointError("YamAdam directional derivative is invalid")

        gradient_views = self._flat_views(gradient, reference_gradients)
        mean_views = self._flat_views(mean, reference_gradients)
        second_views = self._flat_views(second, reference_gradients)
        gamma_views = self._flat_views(gamma, reference_gradients)
        direction_views = self._flat_views(direction, reference_gradients)
        candidates: list[_Candidate] = []
        for index, parameter in enumerate(parameters):
            unused = torch.empty(0, device=gradient.device, dtype=dtype)
            candidates.append(
                _Candidate(
                    parameter=parameter,
                    gradient=gradient_views[index],
                    mean=mean_views[index],
                    second_moment=second_views[index],
                    gamma=gamma_views[index],
                    direction=direction_views[index],
                    old_direction=unused,
                    current_direction=unused,
                    count=counts[index],
                    history_state=None,
                )
            )

        result = _Direction(
            candidates=candidates,
            loss=loss,
            direction_norm=math.sqrt(direction_square_total),
            slope=slope_total,
            directional_gradient=directional_gradient,
            omega=1.0,
            coordinate_count=gradient.numel(),
            gamma_min=gamma_minimum,
            gamma_max=gamma_maximum,
            projection_old_dot=old_dot_total,
            projection_gradient_square=gradient_square_total,
            projection_coefficient=projection_coefficient,
            projection_margin=projection_margin,
            projection_active=projection_coefficient < 0.0,
        )
        views = tuple(
            (candidate.mean, candidate.second_moment, candidate.gamma)
            for candidate in candidates
        )
        self._pending_flat_history = (
            id(result),
            mean,
            second,
            gamma,
            views,
        )
        return result

    def _make_direction_general(
        self,
        raw: list[tuple[Tensor, Tensor, dict]],
        base_loss_tensor: Tensor,
        *,
        previous_step: int,
    ) -> _Direction:
        candidates: list[_Candidate] = []
        first_rows: dict[tuple[torch.device, torch.dtype], list[Tensor]] = {}
        row_parameter_dtypes: dict[
            tuple[torch.device, torch.dtype], list[torch.dtype | None]
        ] = {}
        loss_inserted = False
        direction_arithmetic_epsilon = 0.0

        for parameter, gradient, state in raw:
            reduction_dtype = self._reduction_dtype(gradient)
            direction_arithmetic_epsilon = max(
                direction_arithmetic_epsilon, torch.finfo(gradient.dtype).eps
            )
            gradient_square_tensor = gradient.square()
            has_history = self._history_is_usable(
                parameter, state, previous_step
            )
            if has_history:
                consistency = self._coordinate_consistency(
                    gradient,
                    state["mean"],
                    state["second_moment"],
                    gradient_square=gradient_square_tensor,
                )
                old_gamma = state["gamma"]
                denominator = old_gamma + consistency
                beta = consistency / denominator
                gamma = old_gamma / denominator
                old_numerator = state["mean"] * beta
                mean = old_numerator + gradient * gamma
                second = state["second_moment"] * beta
                second.add_(gradient_square_tensor * gamma)
                second.clamp_min_(0.0)
                torch.maximum(
                    second, mean.square(), out=second
                )
                direction_denominator = second.sqrt()
                nonzero = direction_denominator > 0.0
                zero = torch.zeros_like(direction_denominator)
                old_direction = old_numerator.div(direction_denominator)
                torch.where(nonzero, old_direction, zero, out=old_direction)
                # beta*m_old + gamma*g is the candidate mean, so the
                # unprojected old-plus-current direction is mean/d.  Keeping
                # only this compact form avoids materializing a second full
                # direction tensor; old_direction is retained solely for the
                # global half-space projection scalar below.
                direction = mean.div(direction_denominator)
                torch.where(nonzero, direction, zero, out=direction)
                count = int(state["count"]) + 1
            else:
                mean = gradient.clone()
                second = gradient_square_tensor
                gamma = torch.ones_like(gradient)
                old_direction = torch.zeros_like(gradient)
                # Reuse the exact square used by the original normalization.
                # In particular, if g^2 underflows to zero, the direction is
                # zero rather than sign(g).
                denominator = gradient_square_tensor.sqrt()
                direction = torch.where(
                    denominator > 0.0,
                    gradient / denominator,
                    torch.zeros_like(gradient),
                )
                count = 1

            old_contribution = gradient * old_direction
            old_dot = torch.sum(old_contribution, dtype=reduction_dtype)
            old_absolute_dot = torch.sum(
                old_contribution.abs(), dtype=reduction_dtype
            )
            gradient_square = torch.sum(
                gradient_square_tensor, dtype=reduction_dtype
            )
            if gamma.numel() == 0:
                native_zero = torch.zeros(
                    (), device=gamma.device, dtype=gradient.dtype
                )
                gamma_min = torch.ones_like(native_zero)
                gamma_max = gamma_min.clone()
                mean_min = native_zero
                mean_max = native_zero
                second_min = native_zero
                second_max = native_zero
                direction_min = native_zero
                direction_max = native_zero
            else:
                gamma_min, gamma_max = torch.aminmax(gamma)
                mean_min, mean_max = torch.aminmax(mean)
                second_min, second_max = torch.aminmax(second)
                direction_min, direction_max = torch.aminmax(direction)
            if parameter.numel() == 0:
                parameter_max = torch.zeros(
                    (), device=parameter.device, dtype=gradient.dtype
                )
                gradient_max = parameter_max.clone()
            else:
                parameter_max = (
                    parameter.detach().abs().amax().to(dtype=gradient.dtype)
                )
                gradient_max = gradient.abs().amax()
            loss_value = torch.zeros_like(old_dot)
            if not loss_inserted and base_loss_tensor.device == gradient.device:
                loss_value = base_loss_tensor.detach().to(reduction_dtype).reshape(())
                loss_inserted = True
            bounds = torch.stack(
                (
                    gamma_min,
                    gamma_max,
                    parameter_max,
                    gradient_max,
                    mean_min,
                    mean_max,
                    second_min,
                    second_max,
                    direction_min,
                    direction_max,
                )
            )
            if bounds.dtype != reduction_dtype:
                bounds = bounds.to(reduction_dtype)
            row = torch.cat(
                (
                    torch.stack(
                        (old_dot, gradient_square, old_absolute_dot)
                    ),
                    bounds,
                    loss_value.reshape(1),
                )
            )
            key = (gradient.device, reduction_dtype)
            first_rows.setdefault(key, []).append(row)
            row_parameter_dtypes.setdefault(key, []).append(parameter.dtype)
            unused_component = torch.empty(
                0, device=gradient.device, dtype=gradient.dtype
            )
            candidates.append(
                _Candidate(
                    parameter=parameter,
                    gradient=gradient,
                    mean=mean,
                    second_moment=second,
                    gamma=gamma,
                    direction=direction,
                    old_direction=unused_component,
                    # The production update uses the algebraically compact
                    # direction above.  These fields remain in the private
                    # record only for compatibility with development tools.
                    current_direction=unused_component,
                    count=count,
                    history_state=state if has_history else None,
                )
            )

        if not loss_inserted:
            reduction_dtype = self._reduction_dtype(base_loss_tensor)
            zero = torch.zeros(
                (), device=base_loss_tensor.device, dtype=reduction_dtype
            )
            row = torch.stack(
                (
                    zero,
                    zero,
                    zero,
                    torch.ones_like(zero),
                    torch.ones_like(zero),
                    zero,
                    zero,
                    zero,
                    zero,
                    zero,
                    zero,
                    zero,
                    zero,
                    base_loss_tensor.detach().to(reduction_dtype).reshape(()),
                )
            )
            key = (base_loss_tensor.device, reduction_dtype)
            first_rows.setdefault(key, []).append(row)
            # This row transports only the scalar loss.  It does not
            # correspond to a parameter and must not enter the parameter
            # overflow guard (the closure may even return an integer scalar).
            row_parameter_dtypes.setdefault(key, []).append(None)

        old_dot_total = 0.0
        old_absolute_dot_total = 0.0
        gradient_square_total = 0.0
        gamma_minimum = 1.0
        gamma_maximum = 0.0
        loss = 0.0
        all_valid = True
        for key, device_rows in first_rows.items():
            host_rows = torch.stack(device_rows).detach().cpu().tolist()
            for values, parameter_dtype in zip(
                host_rows, row_parameter_dtypes[key]
            ):
                loss += float(values[13])
                if parameter_dtype is not None:
                    old_dot_total += float(values[0])
                    gradient_square_total += float(values[1])
                    old_absolute_dot_total += float(values[2])
                    gamma_minimum = min(gamma_minimum, float(values[3]))
                    gamma_maximum = max(gamma_maximum, float(values[4]))
                    parameter_maximum = float(values[5])
                    gradient_maximum = float(values[6])
                    all_valid = (
                        all_valid
                        and all(
                            math.isfinite(float(value))
                            for value in values[3:13]
                        )
                        and float(values[3]) > 0.0
                        and float(values[4]) <= 1.0
                        and float(values[9]) >= 0.0
                        and math.isfinite(parameter_maximum)
                        and math.isfinite(gradient_maximum)
                    )

        if (
            not all_valid
            or not math.isfinite(loss)
            or not math.isfinite(old_dot_total)
            or not math.isfinite(old_absolute_dot_total)
            or not math.isfinite(gradient_square_total)
            or old_absolute_dot_total < 0.0
            or gradient_square_total < 0.0
        ):
            raise FloatingPointError(
                "YamAdam encountered a non-finite loss, parameter, gradient, "
                "or moment state"
            )

        projection_margin = 64.0 * direction_arithmetic_epsilon * max(
            abs(old_dot_total), old_absolute_dot_total
        )
        projection_coefficient = (
            min(old_dot_total - projection_margin, 0.0)
            / gradient_square_total
            if gradient_square_total > 0.0
            else 0.0
        )
        if (
            not math.isfinite(projection_margin)
            or projection_margin < 0.0
            or not math.isfinite(projection_coefficient)
        ):
            raise FloatingPointError("YamAdam global projection is invalid")

        second_rows: dict[tuple[torch.device, torch.dtype], list[Tensor]] = {}
        second_bounds: dict[tuple[torch.device, torch.dtype], list[Tensor]] = {}
        for candidate in candidates:
            if projection_coefficient < 0.0:
                if candidate.history_state is not None:
                    # ``mean/d - lambda*g`` is algebraically compact but can
                    # erase a tiny current innovation when the projected old
                    # component cancels in finite precision.  Reconstruct
                    # both components one parameter tensor at a time only on
                    # the active-projection path.  This avoids retaining an
                    # additional full-model tensor on the common path.
                    history = candidate.history_state
                    consistency = self._coordinate_consistency(
                        candidate.gradient,
                        history["mean"],
                        history["second_moment"],
                    )
                    beta = consistency / (history["gamma"] + consistency)
                    denominator = candidate.second_moment.sqrt()
                    nonzero = denominator > 0.0
                    direction = candidate.direction
                    direction.copy_(candidate.gamma).mul_(candidate.gradient)
                    direction.div_(denominator)
                    torch.where(
                        nonzero,
                        direction,
                        torch.zeros_like(direction),
                        out=direction,
                    )
                    old_direction = history["mean"] * beta
                    old_direction.div_(denominator)
                    torch.where(
                        nonzero,
                        old_direction,
                        torch.zeros_like(old_direction),
                        out=old_direction,
                    )
                    old_direction.add_(
                        candidate.gradient, alpha=-projection_coefficient
                    )
                    direction.add_(old_direction)
                else:
                    # A fresh coordinate has no old component, so its compact
                    # direction already is the complete current innovation.
                    direction = candidate.direction
                    # The projection is global: the same correction has
                    # support on fresh coordinates even though their local
                    # old component is zero.
                    direction.add_(
                        candidate.gradient, alpha=-projection_coefficient
                    )
            else:
                direction = candidate.direction
            candidate.direction = direction
            candidate.history_state = None
            reduction_dtype = self._reduction_dtype(candidate.gradient)
            contribution = candidate.gradient * direction
            if direction.numel() == 0:
                native_zero = torch.zeros(
                    (), device=direction.device, dtype=direction.dtype
                )
                direction_min = native_zero
                direction_max = native_zero
                contribution_min = native_zero
                contribution_max = native_zero
            else:
                direction_min, direction_max = torch.aminmax(direction)
                contribution_min, contribution_max = torch.aminmax(contribution)
            finite_bounds = torch.stack(
                (
                    direction_min,
                    direction_max,
                    contribution_min,
                    contribution_max,
                )
            )
            slope = torch.sum(contribution, dtype=reduction_dtype)
            row = torch.stack(
                (
                    slope,
                    torch.sum(contribution.abs(), dtype=reduction_dtype),
                    torch.sum(direction.square(), dtype=reduction_dtype),
                )
            )
            key = (candidate.gradient.device, reduction_dtype)
            second_rows.setdefault(key, []).append(row)
            second_bounds.setdefault(key, []).append(finite_bounds)

        slope_total = 0.0
        absolute_slope_total = 0.0
        direction_square_total = 0.0
        direction_valid = True
        for key, device_rows in second_rows.items():
            metric_totals = torch.stack(device_rows).sum(dim=0)
            finite_bounds = torch.stack(second_bounds[key]).reshape(-1)
            if finite_bounds.dtype != metric_totals.dtype:
                finite_bounds = finite_bounds.to(metric_totals.dtype)
            values = (
                torch.cat((metric_totals, finite_bounds))
                .detach()
                .cpu()
                .tolist()
            )
            slope_total += float(values[0])
            absolute_slope_total += float(values[1])
            direction_square_total += float(values[2])
            direction_valid = direction_valid and all(
                math.isfinite(float(value)) for value in values[3:]
            )
        if (
            not direction_valid
            or not math.isfinite(slope_total)
            or not math.isfinite(absolute_slope_total)
            or not math.isfinite(direction_square_total)
            or direction_square_total < 0.0
        ):
            raise FloatingPointError("YamAdam global direction is non-finite")
        slope_tolerance = 64.0 * direction_arithmetic_epsilon * max(
            abs(old_dot_total),
            abs(projection_coefficient) * gradient_square_total,
            absolute_slope_total,
        )
        if slope_total < -slope_tolerance:
            raise FloatingPointError(
                "YamAdam update direction has negative gradient alignment"
            )
        slope_total = max(0.0, slope_total)

        if slope_total == 0.0:
            omega = 1.0
            directional_gradient = 0.0
        else:
            if direction_square_total <= 0.0:
                raise FloatingPointError("YamAdam produced a zero update direction")
            omega = 1.0
            directional_gradient = slope_total / math.sqrt(direction_square_total)
        if not math.isfinite(directional_gradient):
            raise FloatingPointError("YamAdam directional derivative is invalid")

        return _Direction(
            candidates=candidates,
            loss=loss,
            direction_norm=math.sqrt(direction_square_total),
            slope=slope_total,
            directional_gradient=directional_gradient,
            omega=omega,
            coordinate_count=sum(item.gamma.numel() for item in candidates),
            gamma_min=gamma_minimum,
            gamma_max=gamma_maximum if candidates else 1.0,
            projection_old_dot=old_dot_total,
            projection_gradient_square=gradient_square_total,
            projection_coefficient=projection_coefficient,
            projection_margin=projection_margin,
            projection_active=projection_coefficient < 0.0,
        )

    @staticmethod
    def _buckets(
        candidates: list[_Candidate],
    ) -> list[tuple[list[Tensor], list[Tensor]]]:
        buckets: dict[
            tuple[torch.device, torch.dtype, torch.dtype],
            tuple[list[Tensor], list[Tensor]],
        ] = {}
        for candidate in candidates:
            key = (
                candidate.parameter.device,
                candidate.parameter.dtype,
                candidate.direction.dtype,
            )
            parameters, directions = buckets.setdefault(key, ([], []))
            parameters.append(candidate.parameter)
            directions.append(candidate.direction)
        return list(buckets.values())

    @classmethod
    def _apply_coefficient(
        cls, candidates: list[_Candidate], coefficient: float
    ) -> None:
        """Apply ``x -= coefficient*u`` and undo completed buckets on error."""

        completed: list[tuple[list[Tensor], list[Tensor]]] = []
        with torch.no_grad():
            try:
                for parameters, directions in cls._buckets(candidates):
                    if parameters[0].dtype == directions[0].dtype:
                        torch._foreach_add_(
                            parameters, directions, alpha=-coefficient
                        )
                        completed.append((parameters, directions))
                    else:
                        # Keep the scale factor in the full-precision
                        # direction dtype.  Converting both the direction and
                        # alpha to a low-precision parameter dtype can itself
                        # overflow even when the resulting inward update is
                        # finite.
                        for parameter, direction in zip(parameters, directions):
                            parameter.add_(direction, alpha=-coefficient)
                            completed.append(([parameter], [direction]))
            except BaseException:
                for parameters, directions in reversed(completed):
                    torch._foreach_add_(parameters, directions, alpha=coefficient)
                raise

    @classmethod
    def _restore_and_apply(
        cls,
        candidates: list[_Candidate],
        originals: list[Tensor],
        coefficient: float,
    ) -> None:
        with torch.no_grad():
            for candidate, original in zip(candidates, originals):
                candidate.parameter.copy_(original)
            cls._apply_coefficient(candidates, coefficient)

    @staticmethod
    def _restore_parameters(
        candidates: list[_Candidate], originals: list[Tensor]
    ) -> None:
        with torch.no_grad():
            for candidate, original in zip(candidates, originals):
                candidate.parameter.copy_(original)

    @staticmethod
    def _parameters_changed(
        candidates: list[_Candidate], originals: list[Tensor]
    ) -> bool:
        """Validate parameters and report whether any changed numerically.

        Reductions stay on each parameter device until the final decision, so
        this performs one host synchronization per device rather than one per
        parameter tensor.
        """

        device_checks: dict[torch.device, tuple[list[Tensor], list[Tensor]]] = {}
        for candidate, original in zip(candidates, originals):
            finite_checks, changed_checks = device_checks.setdefault(
                candidate.parameter.device, ([], [])
            )
            finite_checks.append(torch.isfinite(candidate.parameter).all())
            changed_checks.append(
                torch.ne(candidate.parameter, original).any()
            )
        changed = False
        for finite_checks, changed_checks in device_checks.values():
            decision = torch.stack(
                (
                    torch.stack(finite_checks).all(),
                    torch.stack(changed_checks).any(),
                )
            ).detach().cpu().tolist()
            if not bool(decision[0]):
                raise FloatingPointError(
                    "YamAdam update produced a non-finite parameter"
                )
            changed = changed or bool(decision[1])
        return changed

    def _search_initial_armijo_ray(
        self, direction: _Direction, closure: BatchClosure
    ) -> tuple[
        float,
        Tensor,
        float,
        int,
        int,
        int,
        float,
        bool,
        list[Tensor],
    ]:
        """Find a dyadic radius along the unit-L2 direction."""

        if direction.direction_norm <= 0.0 or direction.slope <= 0.0:
            raise FloatingPointError("zero direction cannot initialize YamAdam")
        s = direction.directional_gradient
        originals = [
            item.parameter.detach().clone() for item in direction.candidates
        ]
        trial_radius = 1.0
        best_radius = 0.0
        best_loss = direction.loss
        best_tensor = torch.as_tensor(direction.loss)
        trials = 0
        expansions = 0
        backtracks = 0
        directional_curvature = math.nan
        expansion_blocked = False

        def evaluate(radius: float) -> tuple[bool, bool, float, Tensor]:
            coefficient = radius / direction.direction_norm
            self._restore_and_apply(direction.candidates, originals, coefficient)
            finite = all(
                bool(torch.isfinite(candidate.parameter).all())
                for candidate in direction.candidates
            )
            if not finite:
                return True, False, math.inf, base_tensor
            changed = any(
                not torch.equal(candidate.parameter, original)
                for candidate, original in zip(direction.candidates, originals)
            )
            if not changed:
                # Backtracking can never make a bitwise no-op representable.
                # Let the caller enlarge the radius without paying for a
                # value-only forward.
                return False, False, direction.loss, base_tensor
            with torch.no_grad():
                trial_tensor = closure.value()
            trial_loss = self._loss_value(
                trial_tensor,
                where="initial shift-invariant Armijo trial",
                require_finite=False,
            )
            ratio = (direction.loss - trial_loss) / (radius * s)
            accepted = math.isfinite(ratio) and ratio >= self._ARMIJO_RATIO
            return True, accepted, trial_loss, trial_tensor.detach()

        # Keep the original scalar tensor only as a correctly placed fallback;
        # it is never returned after a successful search.
        base_tensor = torch.as_tensor(
            direction.loss,
            dtype=direction.candidates[0].parameter.dtype,
            device=direction.candidates[0].parameter.device,
        )
        try:
            changed, accepted, trial_loss, trial_tensor = evaluate(trial_radius)
            trials += 1
            while not changed and trials < self._MAX_INITIAL_TRIALS:
                trial_radius *= 2.0
                expansions += 1
                changed, accepted, trial_loss, trial_tensor = evaluate(
                    trial_radius
                )
                trials += 1
            if not changed:
                self._restore_parameters(direction.candidates, originals)
                raise FloatingPointError(
                    "initial Armijo search found no representable radius"
                )
            if accepted:
                best_radius, best_loss, best_tensor = (
                    trial_radius,
                    trial_loss,
                    trial_tensor,
                )
                anchor_result = (best_radius, best_loss, best_tensor)
                loss_curvature = (
                    2.0
                    * (trial_loss - direction.loss)
                    / (trial_radius * trial_radius)
                )
                gradient_curvature = 2.0 * s / trial_radius
                directional_curvature = loss_curvature + gradient_curvature
                # A numerically zero curvature estimate is not enough to
                # identify a truly linear ray: when the base loss is large,
                # the quadratic remainder can be rounded away.  Continue the
                # dyadic search and let a finite rejection or the fixed trial
                # budget decide.  This covers parameter scales far beyond the
                # old 24-step window without adding a loss-scale epsilon.
                expansion_blocked = not math.isfinite(
                    directional_curvature
                )
                if not expansion_blocked:
                    finite_bracket = False
                    while trials < self._MAX_INITIAL_TRIALS:
                        trial_radius *= 2.0
                        changed, accepted, trial_loss, trial_tensor = evaluate(
                            trial_radius
                        )
                        trials += 1
                        if not changed:
                            expansions += 1
                            continue
                        if not accepted:
                            finite_bracket = math.isfinite(trial_loss)
                            break
                        expansions += 1
                        best_radius, best_loss, best_tensor = (
                            trial_radius,
                            trial_loss,
                            trial_tensor,
                        )
                    if not finite_bracket:
                        # Linear and globally concave objectives have no
                        # finite upper Armijo bracket.  Never turn the trial
                        # budget (or an overflow boundary) into a huge
                        # calibration radius: return to the accepted unit
                        # displacement instead.
                        best_radius, best_loss, best_tensor = anchor_result
                        expansion_blocked = True
            else:
                while trials < self._MAX_INITIAL_TRIALS:
                    trial_radius *= 0.5
                    backtracks += 1
                    changed, accepted, trial_loss, trial_tensor = evaluate(
                        trial_radius
                    )
                    trials += 1
                    if not changed:
                        break
                    if accepted:
                        best_radius, best_loss, best_tensor = (
                            trial_radius,
                            trial_loss,
                            trial_tensor,
                        )
                        break
            if best_radius <= 0.0:
                self._restore_parameters(direction.candidates, originals)
                raise FloatingPointError(
                    "initial Armijo search found no finite accepted radius"
                )
            self._restore_and_apply(
                direction.candidates,
                originals,
                best_radius / direction.direction_norm,
            )
            return (
                best_radius,
                best_tensor,
                best_loss,
                trials,
                expansions,
                backtracks,
                directional_curvature,
                expansion_blocked,
                originals,
            )
        except BaseException:
            self._restore_parameters(direction.candidates, originals)
            raise

    def _initial_probe_direction(self, direction: _Direction) -> _Direction:
        """Build the global auxiliary ray used only for calibration.

        Across all active coordinates, the ray is proportional to

        ``g_i / sqrt(g_i**2 + mean(g**2))``.

        The computation first divides by the largest absolute gradient in the
        complete vector.  This avoids adding a dimensional epsilon, makes the
        mathematical ray independent of parameter-tensor packing, and keeps
        it unchanged under a positive rescaling of the complete loss.  The
        ray is never committed: it only supplies a starting radius that is
        checked again on the production direction.
        """

        probe_candidates: list[_Candidate] = []
        metric_groups: dict[
            tuple[torch.device, torch.dtype, torch.dtype],
            list[tuple[Tensor, Tensor, Tensor]],
        ] = {}

        # Concatenating within each device/dtype group makes the reduction
        # order independent of how a fixed flat vector is split into parameter
        # tensors.  Heterogeneous groups meet only as Python double scalars.
        gradient_groups: dict[
            tuple[torch.device, torch.dtype], list[Tensor]
        ] = {}
        coordinate_count = 0
        for candidate in direction.candidates:
            gradient = candidate.gradient
            if gradient.numel() == 0:
                continue
            gradient_groups.setdefault(
                (gradient.device, gradient.dtype), []
            ).append(gradient.reshape(-1))
            coordinate_count += gradient.numel()
        if coordinate_count <= 0:
            raise FloatingPointError(
                "initial auxiliary probe received no active coordinates"
            )

        flat_groups = [
            torch.cat(items) for items in gradient_groups.values()
        ]
        maximum = max(
            float(gradient.detach().abs().amax())
            for gradient in flat_groups
        )
        if not math.isfinite(maximum) or maximum <= 0.0:
            raise FloatingPointError(
                "initial auxiliary probe encountered an invalid gradient"
            )
        normalized_square_total = 0.0
        for gradient in flat_groups:
            scaled_square = (gradient / maximum).square()
            normalized_square_total += float(
                torch.sum(
                    scaled_square,
                    dtype=self._reduction_dtype(scaled_square),
                ).detach()
            )
        mean_square = normalized_square_total / float(coordinate_count)
        if not math.isfinite(mean_square) or mean_square <= 0.0:
            raise FloatingPointError(
                "initial auxiliary probe mean square is invalid"
            )

        for candidate in direction.candidates:
            gradient = candidate.gradient
            if gradient.numel() == 0:
                probe = torch.zeros_like(gradient)
            else:
                scaled = gradient / maximum
                denominator = (scaled.square() + mean_square).sqrt()
                probe = torch.where(
                    denominator > 0.0,
                    scaled / denominator,
                    torch.zeros_like(scaled),
                )

            reduction_dtype = self._reduction_dtype(gradient)
            contribution = gradient * probe
            metric_groups.setdefault(
                (gradient.device, gradient.dtype, reduction_dtype), []
            ).append(
                (
                    contribution.reshape(-1),
                    probe.square().reshape(-1),
                    (candidate.direction * probe).reshape(-1),
                )
            )
            probe_candidates.append(
                _Candidate(
                    parameter=candidate.parameter,
                    gradient=candidate.gradient,
                    mean=candidate.mean,
                    second_moment=candidate.second_moment,
                    gamma=candidate.gamma,
                    direction=probe,
                    old_direction=candidate.old_direction,
                    current_direction=candidate.current_direction,
                    count=candidate.count,
                    history_state=candidate.history_state,
                )
            )

        slope_total = 0.0
        direction_square_total = 0.0
        production_probe_dot = 0.0
        for (_device, _dtype, reduction_dtype), rows in metric_groups.items():
            slope_total += float(
                torch.sum(
                    torch.cat([row[0] for row in rows]),
                    dtype=reduction_dtype,
                ).detach()
            )
            direction_square_total += float(
                torch.sum(
                    torch.cat([row[1] for row in rows]),
                    dtype=reduction_dtype,
                ).detach()
            )
            production_probe_dot += float(
                torch.sum(
                    torch.cat([row[2] for row in rows]),
                    dtype=reduction_dtype,
                ).detach()
            )

        if (
            not math.isfinite(slope_total)
            or not math.isfinite(direction_square_total)
            or slope_total <= 0.0
            or direction_square_total <= 0.0
        ):
            raise FloatingPointError(
                "initial auxiliary probe produced an invalid direction"
            )
        direction_norm = math.sqrt(direction_square_total)
        directional_gradient = slope_total / direction_norm
        cosine = production_probe_dot / (
            direction.direction_norm * direction_norm
        )
        if not math.isfinite(directional_gradient) or not math.isfinite(cosine):
            raise FloatingPointError(
                "initial auxiliary probe produced invalid diagnostics"
            )

        probe_direction = _Direction(
            candidates=probe_candidates,
            loss=direction.loss,
            direction_norm=direction_norm,
            slope=slope_total,
            directional_gradient=directional_gradient,
            omega=direction.omega,
            coordinate_count=direction.coordinate_count,
            gamma_min=direction.gamma_min,
            gamma_max=direction.gamma_max,
            projection_old_dot=0.0,
            projection_gradient_square=direction.projection_gradient_square,
            projection_coefficient=0.0,
            projection_active=False,
            projection_margin=0.0,
        )
        probe_direction.production_cosine = max(-1.0, min(1.0, cosine))
        return probe_direction

    def _initial_armijo(
        self, direction: _Direction, closure: BatchClosure
    ) -> tuple[
        float,
        Tensor,
        float,
        int,
        int,
        int,
        float,
        bool,
        list[Tensor],
    ]:
        """Probe a global ray, then certify the production update."""

        probe_direction = self._initial_probe_direction(direction)
        originals: list[Tensor] | None = None
        try:
            (
                probe_radius,
                _probe_tensor,
                _probe_loss,
                probe_trials,
                probe_expansions,
                probe_backtracks,
                _probe_curvature,
                probe_expansion_blocked,
                originals,
            ) = self._search_initial_armijo_ray(probe_direction, closure)

            if not math.isfinite(probe_radius) or probe_radius <= 0.0:
                raise FloatingPointError(
                    "initial auxiliary probe produced an invalid radius"
                )

            radius = probe_radius
            recheck_trials = 0
            recheck_backtracks = 0
            accepted_tensor: Tensor | None = None
            accepted_loss = math.inf
            while recheck_trials < self._MAX_INITIAL_TRIALS:
                self._restore_and_apply(
                    direction.candidates,
                    originals,
                    radius / direction.direction_norm,
                )
                if not all(
                    bool(torch.isfinite(candidate.parameter).all())
                    for candidate in direction.candidates
                ):
                    accepted = False
                elif not self._parameters_changed(
                    direction.candidates, originals
                ):
                    raise FloatingPointError(
                        "YamAdam update is not representable in the parameter "
                        "dtype; use higher-precision parameters or a "
                        "master-weight setup"
                    )
                else:
                    with torch.no_grad():
                        trial_tensor = closure.value()
                    trial_loss = self._loss_value(
                        trial_tensor,
                        where="initial production-direction recheck",
                        require_finite=False,
                    )
                    ratio = (
                        (direction.loss - trial_loss)
                        / (radius * direction.directional_gradient)
                    )
                    accepted = (
                        math.isfinite(ratio) and ratio >= self._ARMIJO_RATIO
                    )
                    if accepted:
                        accepted_tensor = trial_tensor.detach()
                        accepted_loss = trial_loss
                recheck_trials += 1
                if accepted:
                    break
                radius *= 0.5
                recheck_backtracks += 1

            if accepted_tensor is None:
                raise FloatingPointError(
                    "initial production-direction recheck found no finite "
                    "accepted radius"
                )

            loss_curvature = (
                2.0 * (accepted_loss - direction.loss) / (radius * radius)
            )
            gradient_curvature = (
                2.0 * direction.directional_gradient / radius
            )
            directional_curvature = loss_curvature + gradient_curvature
            limiter = (
                "production_recheck" if radius < probe_radius else "auxiliary_probe"
            )

            group = self.param_groups[0]
            group["yamadam_initial_probe_radius"] = probe_radius
            group["yamadam_initial_probe_trials"] = probe_trials
            group["yamadam_initial_probe_expansions"] = probe_expansions
            group["yamadam_initial_probe_backtracks"] = probe_backtracks
            group["yamadam_initial_probe_directional_gradient"] = (
                probe_direction.directional_gradient
            )
            group["yamadam_initial_probe_cosine"] = float(
                probe_direction.production_cosine
            )
            group["yamadam_initial_recheck_trials"] = recheck_trials
            group["yamadam_initial_recheck_backtracks"] = recheck_backtracks
            group["yamadam_initial_radius_limiter"] = limiter

            return (
                radius,
                accepted_tensor,
                accepted_loss,
                probe_trials + recheck_trials,
                probe_expansions,
                probe_backtracks + recheck_backtracks,
                directional_curvature,
                probe_expansion_blocked or radius < probe_radius,
                originals,
            )
        except BaseException:
            if originals is not None:
                self._restore_parameters(direction.candidates, originals)
            raise

    def _snapshot_metadata(
        self,
        *_unused_candidates: object,
    ) -> tuple[
        dict[str, object],
        list[tuple[Tensor, bool, dict]],
        YamAdamStepInfo | None,
        dict[str, object] | None,
    ]:
        group = self.param_groups[0]
        group_snapshot = {key: value for key, value in group.items() if key != "params"}
        state_snapshot = [
            (
                parameter,
                parameter in self.state,
                dict(self.state.get(parameter, {})),
            )
            for parameter in group["params"]
        ]
        return (
            group_snapshot,
            state_snapshot,
            self._last_info,
            self._flat_history_cache,
        )

    def _restore_metadata(
        self,
        snapshot: tuple[
            dict[str, object],
            list[tuple[Tensor, bool, dict]],
            YamAdamStepInfo | None,
            dict[str, object] | None,
        ],
    ) -> None:
        group_snapshot, state_snapshot, last_info, flat_history_cache = snapshot
        group = self.param_groups[0]
        for key in list(group):
            if key != "params":
                del group[key]
        group.update(group_snapshot)
        for parameter, existed, saved in state_snapshot:
            if existed:
                current = self.state[parameter]
                current.clear()
                current.update(saved)
            else:
                self.state.pop(parameter, None)
        self._last_info = last_info
        self._flat_history_cache = flat_history_cache
        # A pending buffer describes the failed candidate, never committed
        # state.  Restoring it would retain an orphaned full-model buffer.
        self._pending_flat_history = None

    def _commit_direction(self, direction: _Direction, *, step_number: int) -> None:
        pending = self._pending_flat_history
        try:
            for item in direction.candidates:
                state = self.state[item.parameter]
                state["mean"] = item.mean
                state["second_moment"] = item.second_moment
                state["gamma"] = item.gamma
                state["count"] = item.count
                state["last_seen_step"] = step_number
            gamma_bytes = 0
            for parameter in self.param_groups[0]["params"]:
                gamma = self.state.get(parameter, {}).get("gamma")
                if isinstance(gamma, Tensor):
                    gamma_bytes += gamma.numel() * gamma.element_size()
            group = self.param_groups[0]
            group["yamadam_steps"] = step_number
            group["yamadam_last_s"] = direction.directional_gradient
            group["yamadam_last_omega"] = direction.omega
            group["yamadam_coordinate_count"] = direction.coordinate_count
            group["yamadam_gamma_min"] = direction.gamma_min
            group["yamadam_gamma_max"] = direction.gamma_max
            group["yamadam_gamma_bytes"] = gamma_bytes
            group["yamadam_projection_old_dot"] = direction.projection_old_dot
            group["yamadam_projection_gradient_square"] = (
                direction.projection_gradient_square
            )
            group["yamadam_projection_coefficient"] = (
                direction.projection_coefficient
            )
            group["yamadam_projection_margin"] = direction.projection_margin
            group["yamadam_projection_active"] = direction.projection_active

            if pending is not None and pending[0] == id(direction):
                self._flat_history_cache = {
                    "step": step_number,
                    "buffers": pending[1:4],
                    "views": pending[4],
                }
            else:
                # The general path may update only parameters with gradients.
                # Clone any inactive states that still view the old flat
                # buffers before dropping cache ownership.  This is inside the
                # metadata transaction, so a clone or later commit failure
                # restores the exact old state tensors and cache object.
                self._detach_flat_history()
        finally:
            # The direction candidates themselves keep buffers alive until the
            # caller returns; this additional transaction-local owner is never
            # needed after a commit attempt, successful or otherwise.
            self._pending_flat_history = None

    @staticmethod
    def _parameter_copies(direction: _Direction) -> list[Tensor]:
        """Copy active raw parameters for exact transactional rollback."""

        return [
            item.parameter.detach().clone() for item in direction.candidates
        ]

    def _steady_parameter_copies(self, direction: _Direction) -> list[Tensor]:
        """Create the rollback journal, reusing flat-path scratch storage.

        The flat path owns a concatenated gradient buffer after all direction
        and scale statistics have been computed.  A normal post-initialization
        update no longer needs that buffer, so its views can hold exact copies
        of the parameters instead of allocating another full-model buffer.
        General, partial-gradient, mixed-dtype, and initial-search paths retain
        the ordinary clone-based journal.
        """

        pending = self._pending_flat_history
        if pending is None or pending[0] != id(direction):
            return self._parameter_copies(direction)
        originals = [item.gradient for item in direction.candidates]
        parameters = [item.parameter.detach() for item in direction.candidates]
        with torch.no_grad():
            torch._foreach_copy_(originals, parameters)
        return originals

    def _steady_parameters_changed(
        self, direction: _Direction, originals: list[Tensor]
    ) -> bool:
        """Validate parameters and check numerical representability.

        On the flat path the normalized direction is not used after applying
        the update.  Its views therefore form a second scratch vector for one
        foreach subtraction and infinity-norm reduction.  The rollback journal
        remains untouched, so every later failure can still restore parameters
        exactly.  Other paths use the device-grouped reference check.
        """

        pending = self._pending_flat_history
        if pending is None or pending[0] != id(direction):
            return self._parameters_changed(direction.candidates, originals)
        differences = [item.direction for item in direction.candidates]
        parameters = [item.parameter.detach() for item in direction.candidates]
        with torch.no_grad():
            torch._foreach_copy_(differences, parameters)
            torch._foreach_sub_(differences, originals)
            parameter_bounds = torch._foreach_norm(parameters, float("inf"))
            maxima = torch._foreach_norm(differences, float("inf"))
        decision = torch.stack(
            (
                torch.stack(parameter_bounds).isfinite().all(),
                torch.stack(maxima).ne(0.0).any(),
            )
        ).detach().cpu().tolist()
        if not bool(decision[0]):
            raise FloatingPointError("YamAdam update produced a non-finite parameter")
        return bool(decision[1])

    @staticmethod
    def _finish_replay(closure: BatchClosure) -> None:
        if isinstance(closure, ReplayableBatchClosure):
            closure._restore_base_forward()

    def _load_training_state_dict(self, state_dict: dict) -> None:
        """Atomically load raw state while retaining full-precision tensors."""

        saved_groups = state_dict.get("param_groups", [])
        saved_state = state_dict.get("state", {})
        saved_group = saved_groups[0]
        saved_ids = self._validate_serialized_parameter_ids(
            saved_group.get("params"),
            saved_state,
            expected_count=len(self.param_groups[0]["params"]),
        )

        if "yamadam_initialized" not in saved_group:
            raise ValueError("checkpoint is missing yamadam_initialized")
        initialized = saved_group["yamadam_initialized"]
        if type(initialized) is not bool:
            raise ValueError("checkpoint yamadam_initialized must be a bool")

        if "yamadam_steps" not in saved_group:
            raise ValueError("checkpoint is missing yamadam_steps")
        steps = saved_group["yamadam_steps"]
        if not isinstance(steps, int) or isinstance(steps, bool) or steps < 0:
            raise ValueError(
                "checkpoint yamadam_steps must be a non-negative integer"
            )
        if initialized != (steps > 0):
            raise ValueError(
                "checkpoint initialization must be false exactly when "
                "yamadam_steps is zero"
            )

        saved_scale = saved_group.get("yamadam_scale_name")
        if saved_scale != self._SCALE_POLICY.name:
            raise ValueError("optimizer state uses a different scale rule")

        tensor_names = ("mean", "second_moment", "gamma")
        history_names = frozenset(
            (*tensor_names, "count", "last_seen_step")
        )
        prepared: list[tuple[Tensor, str, Tensor]] = []
        history_last_seen: list[int] = []
        current_parameters = self.param_groups[0]["params"]
        for parameter, saved_id in zip(current_parameters, saved_ids):
            if saved_id not in saved_state:
                continue
            source = saved_state[saved_id]
            if not isinstance(source, Mapping):
                raise ValueError(
                    "checkpoint parameter state must be an empty or complete "
                    "mapping"
                )
            names = frozenset(source.keys())
            if not names:
                continue
            if names != history_names:
                missing = sorted(history_names.difference(names))
                unexpected = sorted(
                    (repr(name) for name in names.difference(history_names))
                )
                details: list[str] = []
                if missing:
                    details.append(f"missing {missing!r}")
                if unexpected:
                    details.append(f"unexpected {unexpected!r}")
                raise ValueError(
                    "checkpoint parameter history must contain exactly "
                    f"{sorted(history_names)!r} ({'; '.join(details)})"
                )

            expected_dtype = self._state_dtype(parameter)
            for name in tensor_names:
                value = source[name]
                if not isinstance(value, Tensor):
                    raise ValueError(
                        f"optimizer tensor state {name!r} must be a Tensor"
                    )
                if value.shape != parameter.shape:
                    raise ValueError(
                        f"optimizer tensor state {name!r} has wrong shape"
                    )
                if value.dtype != expected_dtype:
                    raise ValueError(
                        f"optimizer tensor state {name!r} has wrong dtype"
                    )
                try:
                    finite = bool(torch.isfinite(value).all().item())
                except Exception as error:
                    raise ValueError(
                        f"optimizer tensor state {name!r} cannot be "
                        "validated for finiteness"
                    ) from error
                if not finite:
                    raise ValueError(
                        f"optimizer tensor state {name!r} must be finite"
                    )
                prepared.append((parameter, name, value))

            count = source["count"]
            last_seen_step = source["last_seen_step"]
            if (
                not isinstance(count, int)
                or isinstance(count, bool)
                or count < 1
            ):
                raise ValueError(
                    "checkpoint history count must be a positive integer"
                )
            if (
                not isinstance(last_seen_step, int)
                or isinstance(last_seen_step, bool)
            ):
                raise ValueError(
                    "checkpoint history last_seen_step must be an integer"
                )
            if not 1 <= count <= last_seen_step <= steps:
                raise ValueError(
                    "checkpoint history must satisfy "
                    "1 <= count <= last_seen_step <= yamadam_steps"
                )
            history_last_seen.append(last_seen_step)

        if steps == 0 and history_last_seen:
            raise ValueError("a zero-step checkpoint cannot contain history")
        if steps > 0 and (
            not history_last_seen or max(history_last_seen) != steps
        ):
            raise ValueError(
                "checkpoint history does not reach the saved yamadam_steps"
            )

        previous_state = self.state
        previous_groups = self.param_groups
        previous_defaults = dict(self.defaults)
        previous_info = self._last_info
        previous_flat_cache = self._flat_history_cache
        previous_pending_cache = self._pending_flat_history
        try:
            super().load_state_dict(state_dict)
            self._ensure_group_state()
            for parameter, name, value in prepared:
                self.state[parameter][name] = (
                    value.detach()
                    .to(
                        device=parameter.device,
                        dtype=self._state_dtype(parameter),
                    )
                    .clone()
                )
            self._last_info = None
            # Serialized per-parameter tensors own independent storage.  The
            # next eligible step may flatten them and establish a new cache.
            self._flat_history_cache = None
            self._pending_flat_history = None
        except BaseException:
            self.state = previous_state
            self.param_groups = previous_groups
            self.defaults.clear()
            self.defaults.update(previous_defaults)
            self._last_info = previous_info
            self._flat_history_cache = previous_flat_cache
            self._pending_flat_history = previous_pending_cache
            raise

    @staticmethod
    def _validate_serialized_parameter_ids(
        saved_ids: Any,
        saved_state: Mapping[Any, Any],
        *,
        expected_count: int,
    ) -> tuple[int, ...]:
        """Reject IDs that PyTorch would otherwise zip ambiguously."""

        if (
            not isinstance(saved_ids, Sequence)
            or isinstance(saved_ids, (str, bytes))
            or len(saved_ids) != expected_count
        ):
            raise ValueError(
                "checkpoint parameter count does not match optimizer"
            )
        identifiers = tuple(saved_ids)
        if any(
            type(identifier) is not int or identifier < 0
            for identifier in identifiers
        ):
            raise ValueError(
                "checkpoint parameter IDs must be non-negative integers"
            )
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("checkpoint parameter IDs must be unique")

        state_keys = tuple(saved_state.keys())
        if any(type(key) is not int or key < 0 for key in state_keys):
            raise ValueError(
                "checkpoint state keys must be non-negative integers"
            )
        identifier_set = set(identifiers)
        state_key_set = set(state_keys)
        if not state_key_set.issubset(identifier_set):
            raise ValueError(
                "checkpoint optimizer state keys do not match parameter IDs"
            )
        return identifiers

    def load_state_dict(self, state_dict: dict) -> None:
        """Load raw state, migrating legacy training checkpoints atomically."""

        if not isinstance(state_dict, Mapping):
            raise TypeError("optimizer state_dict must be a mapping")
        saved_groups = state_dict.get("param_groups")
        saved_state = state_dict.get("state")
        if (
            not isinstance(saved_groups, Sequence)
            or isinstance(saved_groups, (str, bytes))
            or len(saved_groups) != 1
            or not isinstance(saved_groups[0], Mapping)
        ):
            raise ValueError("checkpoint must contain one parameter group")
        if not isinstance(saved_state, Mapping):
            raise ValueError("checkpoint has invalid optimizer state")

        saved_group = saved_groups[0]
        marker = saved_group.get("yamadam_average_eval_mode", False)
        if type(marker) is not bool:
            raise ValueError(
                "legacy checkpoint evaluation-mode marker must be a bool"
            )
        if marker:
            raise ValueError(
                "raw-only optimizer cannot load an evaluation-mode checkpoint; "
                "resume from a training-mode checkpoint"
            )

        legacy_group_names = (
            "yamadam_average_name",
            "yamadam_average_count",
            "yamadam_average_bytes",
            "yamadam_average_eval_mode",
        )
        cleaned_group = dict(saved_group)
        for name in legacy_group_names:
            cleaned_group.pop(name, None)

        cleaned_state: dict[Any, Any] = {}
        for key, value in saved_state.items():
            if not isinstance(value, Mapping):
                cleaned_state[key] = value
                continue
            if "yamadam_raw_backup" in value:
                raise ValueError(
                    "raw-only optimizer rejects a checkpoint containing an "
                    "evaluation raw-parameter backup"
                )
            cleaned = dict(value)
            cleaned.pop("yamadam_output_average", None)
            cleaned_state[key] = cleaned

        transport = dict(state_dict)
        transport["param_groups"] = [cleaned_group]
        transport["state"] = cleaned_state
        self._load_training_state_dict(transport)

    def state_dict(self) -> dict:
        """Serialize only raw parameters' coordinate and scale state."""

        result = super().state_dict()
        forbidden_group = {
            "yamadam_average_name",
            "yamadam_average_count",
            "yamadam_average_bytes",
            "yamadam_average_eval_mode",
        }
        if any(
            forbidden_group.intersection(group)
            for group in result["param_groups"]
        ):
            raise RuntimeError(
                "raw-only optimizer contains output-average metadata"
            )
        for value in result["state"].values():
            if isinstance(value, Mapping) and (
                "yamadam_output_average" in value
                or "yamadam_raw_backup" in value
            ):
                raise RuntimeError(
                    "raw-only optimizer contains output-average state"
                )
        return result

    def eval(self) -> None:
        """Compatibility no-op: the raw iterate is the evaluation output."""

        return None

    def train(self) -> None:
        """Compatibility no-op matching :meth:`eval`."""

        return None

    def _initialize_scale_proposal(
        self,
        direction: _Direction,
        *,
        calibration_radius: float,
    ) -> ScaleProposal:
        """Return the first committed scale proposal.

        The hook keeps the public optimizer's behavior unchanged while letting
        development-only scale experiments use information from the complete
        direction without copying the transactional :meth:`step` machinery.
        """

        return self._SCALE_POLICY.initialize(
            calibration_radius=calibration_radius,
            s=direction.directional_gradient,
            omega=direction.omega,
        )

    def _steady_scale_proposal(self, direction: _Direction) -> ScaleProposal:
        """Return a post-calibration scale proposal for ``direction``."""

        return self._SCALE_POLICY.propose(
            self.param_groups[0],
            s=direction.directional_gradient,
            omega=direction.omega,
        )

    def step(self, closure: BatchClosure | None = None) -> Tensor:
        """Take one optimization step.

        The first nonzero step searches a global pooled-gradient probe for a
        candidate radius, then applies Armijo backtracking to that radius on
        ``w=u/||u||`` before committing the production direction.  Later
        radii follow the selected scale policy.  In the public final policy
        they never exceed ``min(C, 2*C/sqrt(k))`` and remain coupled to the
        current directional derivative relative to its cumulative RMS
        history.  Zero directions do not advance ``k``.

        Thus, after initialization, the closure is evaluated exactly once and
        its loss value does not enter the step-scale rule.
        """

        if closure is None:
            raise TypeError("YamAdam.step requires a BatchClosure")
        if not callable(getattr(closure, "value_and_grad", None)) or not callable(
            getattr(closure, "value", None)
        ):
            raise TypeError(
                "closure must provide value_and_grad() and value() methods"
            )

        self.zero_grad(set_to_none=True)
        with torch.enable_grad():
            if (
                type(closure) is ReplayableBatchClosure
                and bool(self.param_groups[0]["yamadam_initialized"])
            ):
                base_loss_tensor = closure._yamadam_value_and_grad_no_replay()
            else:
                base_loss_tensor = closure.value_and_grad()
        replay_closed = False
        try:
            self._validate_loss_tensor(base_loss_tensor, where="base evaluation")
            group = self.param_groups[0]
            previous_step = int(group["yamadam_steps"])
            raw = self._raw_gradients()
            if not raw:
                base_loss = self._loss_value(
                    base_loss_tensor, where="base evaluation"
                )
                self._finish_replay(closure)
                replay_closed = True
                self._last_info = YamAdamStepInfo(
                    accepted=False,
                    reason="no_gradient",
                    loss_before=base_loss,
                    loss_after=base_loss,
                    radius=0.0,
                    directional_derivative=0.0,
                    omega=1.0,
                    scale=dict(self._SCALE_POLICY.diagnostics(group)),
                    projection_old_dot=float(
                        group["yamadam_projection_old_dot"]
                    ),
                    projection_coefficient=float(
                        group["yamadam_projection_coefficient"]
                    ),
                    projection_margin=float(
                        group["yamadam_projection_margin"]
                    ),
                    projection_active=bool(
                        group["yamadam_projection_active"]
                    ),
                    trials=0,
                    expansions=0,
                    backtracks=0,
                )
                return base_loss_tensor.detach()

            direction = self._make_direction(
                raw, base_loss_tensor, previous_step=previous_step
            )
            next_step = previous_step + 1
            snapshot = self._snapshot_metadata()

            if direction.slope == 0.0:
                self._finish_replay(closure)
                replay_closed = True
                # A zero direction carries no optimization evidence.  Do not
                # create coordinate history or advance the global clock: an
                # uninitialized optimizer must remain a valid zero-step
                # checkpoint, and an initialized optimizer must retain the
                # last committed history until a nonzero direction arrives.
                self._pending_flat_history = None
                group["yamadam_last_rho"] = 0.0
                self._last_info = YamAdamStepInfo(
                    accepted=False,
                    reason="zero_direction",
                    loss_before=direction.loss,
                    loss_after=direction.loss,
                    radius=0.0,
                    directional_derivative=0.0,
                    omega=direction.omega,
                    scale=dict(self._SCALE_POLICY.diagnostics(group)),
                    projection_old_dot=direction.projection_old_dot,
                    projection_coefficient=direction.projection_coefficient,
                    projection_margin=direction.projection_margin,
                    projection_active=direction.projection_active,
                    trials=0,
                    expansions=0,
                    backtracks=0,
                )
                return base_loss_tensor.detach()

            directional_square = direction.directional_gradient**2
            if not math.isfinite(directional_square) or directional_square <= 0.0:
                raise FloatingPointError("YamAdam directional square overflowed")

            if not bool(group["yamadam_initialized"]):
                (
                    radius,
                    accepted_tensor,
                    accepted_loss,
                    trials,
                    expansions,
                    backtracks,
                    initial_curvature,
                    expansion_blocked,
                    originals,
                ) = self._initial_armijo(direction, closure)
                try:
                    scale_proposal = self._initialize_scale_proposal(
                        direction,
                        calibration_radius=radius,
                    )
                    self._finish_replay(closure)
                    replay_closed = True
                    self._commit_direction(direction, step_number=next_step)
                    group["yamadam_initialized"] = True
                    group["yamadam_radius"] = radius
                    group.update(scale_proposal.updates)
                    group["yamadam_last_rho"] = scale_proposal.radius
                    group["yamadam_initial_trials"] = trials
                    group["yamadam_initial_expansions"] = expansions
                    group["yamadam_initial_backtracks"] = backtracks
                    group["yamadam_initial_curvature"] = initial_curvature
                    group["yamadam_initial_expansion_blocked"] = expansion_blocked
                except BaseException:
                    self._restore_parameters(direction.candidates, originals)
                    self._restore_metadata(snapshot)
                    raise
                self._last_info = YamAdamStepInfo(
                    accepted=True,
                    reason="initial_armijo",
                    loss_before=direction.loss,
                    loss_after=accepted_loss,
                    radius=radius,
                    directional_derivative=direction.directional_gradient,
                    omega=direction.omega,
                    scale=dict(self._SCALE_POLICY.diagnostics(group)),
                    projection_old_dot=direction.projection_old_dot,
                    projection_coefficient=direction.projection_coefficient,
                    projection_margin=direction.projection_margin,
                    projection_active=direction.projection_active,
                    trials=trials,
                    expansions=expansions,
                    backtracks=backtracks,
                )
                return accepted_tensor

            scale_proposal = self._steady_scale_proposal(direction)
            radius = scale_proposal.radius
            if not math.isfinite(radius) or radius <= 0.0:
                raise FloatingPointError("YamAdam step-scale state is invalid")
            coefficient = radius / direction.direction_norm
            if not math.isfinite(coefficient) or coefficient <= 0.0:
                raise FloatingPointError("YamAdam update coefficient is invalid")
            originals = self._steady_parameter_copies(direction)

            # Close a replay interval before committing any optimizer/model
            # mutation.  Candidate state is out of place, so all data-dependent
            # failures above leave the committed state untouched.
            self._finish_replay(closure)
            replay_closed = True
            try:
                self._apply_coefficient(direction.candidates, coefficient)
                if not self._steady_parameters_changed(direction, originals):
                    raise FloatingPointError(
                        "YamAdam update is not representable in the parameter "
                        "dtype; use higher-precision parameters or a "
                        "master-weight setup"
                    )
                self._commit_direction(direction, step_number=next_step)
                group.update(scale_proposal.updates)
                group["yamadam_last_rho"] = radius
            except BaseException:
                self._restore_parameters(direction.candidates, originals)
                self._restore_metadata(snapshot)
                raise

            self._last_info = YamAdamStepInfo(
                accepted=True,
                reason="guarded_scale_update",
                loss_before=direction.loss,
                loss_after=direction.loss,
                radius=radius,
                directional_derivative=direction.directional_gradient,
                omega=direction.omega,
                scale=dict(self._SCALE_POLICY.diagnostics(group)),
                projection_old_dot=direction.projection_old_dot,
                projection_coefficient=direction.projection_coefficient,
                projection_margin=direction.projection_margin,
                projection_active=direction.projection_active,
                trials=0,
                expansions=0,
                backtracks=0,
            )
            return base_loss_tensor.detach()
        except BaseException:
            if not replay_closed:
                self._finish_replay(closure)
            raise
        finally:
            # Candidate buffers are transaction-local unless _commit_direction
            # promoted them to _flat_history_cache.  In particular, Armijo,
            # representability, commit, and averaging failures must not leave
            # an orphaned full-model buffer owned by the optimizer.
            self._pending_flat_history = None


from dataclasses import replace



class _CappedMaxGuardedScale(_GuardedInverseSqrtScale):
    """Use the larger base radius, capped at twice the smaller one."""

    name = "capped_max_guarded_inverse_sqrt"

    @classmethod
    def initialize(
        cls, *, calibration_radius: float, s: float, omega: float
    ) -> ScaleProposal:
        proposal = super().initialize(
            calibration_radius=calibration_radius,
            s=s,
            omega=omega,
        )
        updates = dict(proposal.updates)
        updates["yamadam_scale_name"] = cls.name
        return replace(proposal, updates=updates)

    @classmethod
    def propose(cls, group: dict, *, s: float, omega: float) -> ScaleProposal:
        proposal = super().propose(group, s=s, omega=omega)
        clock = float(proposal.updates["yamadam_scale_clock_radius"])
        relative = float(proposal.updates["yamadam_scale_relative_radius"])
        smaller = min(clock, relative)
        larger = max(clock, relative)
        radius = min(larger, 2.0 * smaller)
        if (
            not math.isfinite(radius)
            or radius < smaller
            or radius > larger
            or radius > 2.0 * smaller
        ):
            raise FloatingPointError("capped-max guarded radius is invalid")
        updates = dict(proposal.updates)
        updates["yamadam_scale_name"] = cls.name
        return ScaleProposal(radius=radius, updates=updates)


class _NeutralMemoryYamAdam(_YamAdamCore):
    """Parameter-only optimizer with neutral memory and a capped-max radius."""

    _SCALE_POLICY = _CappedMaxGuardedScale

    @classmethod
    def _coordinate_consistency(
        cls,
        gradient: Tensor,
        old_mean: Tensor,
        old_second_moment: Tensor,
        *,
        gradient_square: Tensor | None = None,
    ) -> Tensor:
        del cls, old_mean, old_second_moment, gradient_square
        return torch.full_like(gradient, 0.5)


class _PreviousAgreementGeometricSupportBridgeScale(
    _CappedMaxGuardedScale
):
    """Apply a support-safe geometric release from committed evidence."""

    name = (
        "previous_agreement_geometric_support_bridge_"
        "projection_primed_clock_capped_max"
    )
    _PROJECTION_SEEN_KEY = "yamadam_projection_seen"

    @classmethod
    def _projection_seen(cls, group: dict) -> bool:
        seen = group.get(cls._PROJECTION_SEEN_KEY, False)
        if type(seen) is not bool:
            raise ValueError("projection-seen latch must be a bool")
        return seen

    @classmethod
    def ensure_group(cls, group: dict) -> None:
        super().ensure_group(group)
        group.setdefault(cls._PROJECTION_SEEN_KEY, False)
        cls._projection_seen(group)

    @classmethod
    def initialize(
        cls, *, calibration_radius: float, s: float, omega: float
    ) -> ScaleProposal:
        proposal = super().initialize(
            calibration_radius=calibration_radius,
            s=s,
            omega=omega,
        )
        updates = dict(proposal.updates)
        updates["yamadam_scale_name"] = cls.name
        return replace(proposal, updates=updates)

    @staticmethod
    def _capped_max(*, clock: float, relative: float) -> float:
        if (
            not math.isfinite(clock)
            or not math.isfinite(relative)
            or clock <= 0.0
            or relative <= 0.0
        ):
            raise FloatingPointError("geometric clock radii are invalid")
        smaller = min(clock, relative)
        larger = max(clock, relative)
        doubled = 2.0 * smaller
        if not math.isfinite(doubled):
            doubled = larger
        radius = min(larger, doubled)
        if not math.isfinite(radius) or radius < smaller or radius > larger:
            raise FloatingPointError("geometric capped-max radius is invalid")
        return radius

    @staticmethod
    def _geometric_clock(*, calibration_radius: float, clock: float) -> float:
        """Return ``sqrt(C) * sqrt(clock)`` with finite endpoint guards."""

        if (
            not math.isfinite(calibration_radius)
            or not math.isfinite(clock)
            or calibration_radius <= 0.0
            or clock <= 0.0
        ):
            raise FloatingPointError("geometric clock inputs are invalid")
        lower = min(calibration_radius, clock)
        upper = max(calibration_radius, clock)
        geometric = math.sqrt(calibration_radius) * math.sqrt(clock)
        if not math.isfinite(geometric):
            geometric = upper
        geometric = min(upper, max(lower, geometric))
        if not math.isfinite(geometric) or geometric <= 0.0:
            raise FloatingPointError("geometric clock is invalid")
        return geometric

    @classmethod
    def _target_radius(
        cls,
        *,
        calibration_radius: float,
        clock: float,
        relative: float,
        production_radius: float,
    ) -> tuple[float, float, float]:
        geometric_clock = cls._geometric_clock(
            calibration_radius=calibration_radius,
            clock=clock,
        )
        geometric_radius = cls._capped_max(
            clock=geometric_clock,
            relative=relative,
        )
        doubled_production = 2.0 * production_radius
        target = (
            geometric_radius
            if not math.isfinite(doubled_production)
            else min(geometric_radius, doubled_production)
        )
        tolerance = 8.0 * math.ulp(max(1.0, production_radius, target))
        if (
            target < production_radius
            and production_radius - target <= tolerance
        ):
            target = production_radius
        if (
            not math.isfinite(production_radius)
            or production_radius <= 0.0
            or not math.isfinite(target)
            or target < production_radius
            or target > geometric_radius
            or target > 2.0 * relative
            or (
                math.isfinite(doubled_production)
                and target > doubled_production
            )
        ):
            raise FloatingPointError("geometric target radius is invalid")
        return geometric_clock, geometric_radius, target

    @staticmethod
    def _geometric_support_release(
        *,
        agreement: float,
        projection_seen: bool,
        target_gap: float,
        support_debt: float,
    ) -> float:
        """Return ``sqrt(G) * sqrt([h*a*G-D]_+)`` with ``h=projection_seen``."""

        if (
            not math.isfinite(agreement)
            or agreement < 0.0
            or agreement > 1.0
        ):
            raise FloatingPointError(
                "geometric-support agreement is outside [0, 1]"
            )
        if type(projection_seen) is not bool:
            raise ValueError("projection-seen latch must be a bool")
        if (
            not math.isfinite(target_gap)
            or target_gap < 0.0
            or not math.isfinite(support_debt)
            or support_debt < 0.0
        ):
            raise FloatingPointError(
                "geometric-support radius inputs are invalid"
            )
        if not projection_seen or agreement == 0.0 or target_gap == 0.0:
            return 0.0

        support_release = max(
            0.0,
            agreement * target_gap - support_debt,
        )
        if support_release == 0.0:
            return 0.0

        release = math.sqrt(target_gap) * math.sqrt(support_release)
        release = min(target_gap, max(support_release, release))
        if (
            not math.isfinite(release)
            or release <= 0.0
            or release > target_gap
            or release < support_release
        ):
            raise FloatingPointError(
                "geometric-support bridge release is invalid"
            )
        return release

    @classmethod
    def _release_enabled(cls, group: dict) -> bool:
        previous_agreement = float(group.get("yamadam_last_omega", 0.0))
        if (
            not math.isfinite(previous_agreement)
            or previous_agreement < 0.0
            or previous_agreement > 1.0
        ):
            raise FloatingPointError(
                "previous geometric-support agreement is outside [0, 1]"
            )
        return cls._projection_seen(group) and previous_agreement > 0.0

    @classmethod
    def propose(cls, group: dict, *, s: float, omega: float) -> ScaleProposal:
        agreement = float(omega)
        if (
            not math.isfinite(agreement)
            or agreement < 0.0
            or agreement > 1.0
        ):
            raise FloatingPointError(
                "geometric-support agreement is outside [0, 1]"
            )

        production = _CappedMaxGuardedScale.propose(
            group,
            s=s,
            omega=agreement,
        )
        updates = dict(production.updates)
        clock = float(updates["yamadam_scale_clock_radius"])
        relative = float(updates["yamadam_scale_relative_radius"])
        calibration_radius = float(group["yamadam_radius"])
        _q, _geometric, target = cls._target_radius(
            calibration_radius=calibration_radius,
            clock=clock,
            relative=relative,
            production_radius=production.radius,
        )
        target_gap = max(0.0, target - production.radius)
        support_debt = max(0.0, production.radius - relative)
        release = cls._geometric_support_release(
            agreement=agreement,
            projection_seen=cls._release_enabled(group),
            target_gap=target_gap,
            support_debt=support_debt,
        )
        radius = production.radius + release
        radius = min(target, max(production.radius, radius))

        doubled_production = 2.0 * production.radius
        if (
            not math.isfinite(radius)
            or radius < production.radius
            or radius > target
            or radius > 2.0 * relative
            or (
                math.isfinite(doubled_production)
                and radius > doubled_production
            )
        ):
            raise FloatingPointError(
                "geometric-support bridge radius is invalid"
            )
        updates["yamadam_scale_name"] = cls.name
        return ScaleProposal(radius=radius, updates=updates)

    @classmethod
    def diagnostics(cls, group: dict) -> dict[str, bool | float | str]:
        result: dict[str, bool | float | str] = dict(
            super().diagnostics(group)
        )
        result["projection_seen"] = cls._projection_seen(group)
        return result


class _ReleaseYamAdam(_NeutralMemoryYamAdam):
    _SCALE_POLICY = _PreviousAgreementGeometricSupportBridgeScale

    @staticmethod
    def _projection_agreement(direction: _Direction) -> float:
        old_dot = float(direction.projection_old_dot)
        margin = float(direction.projection_margin)
        direction_epsilon = max(
            (
                torch.finfo(candidate.gradient.dtype).eps
                for candidate in direction.candidates
            ),
            default=0.0,
        )
        if (
            not math.isfinite(old_dot)
            or not math.isfinite(margin)
            or margin < 0.0
            or not math.isfinite(direction_epsilon)
            or direction_epsilon < 0.0
        ):
            raise FloatingPointError("projection agreement inputs are invalid")
        if old_dot <= margin or margin == 0.0:
            return 0.0
        if direction_epsilon == 0.0:
            raise FloatingPointError("projection agreement dtype is invalid")

        encoded_absolute_dot = margin / (64.0 * direction_epsilon)
        if not math.isfinite(encoded_absolute_dot):
            return 0.0
        if encoded_absolute_dot <= 0.0:
            return 0.0
        agreement = (old_dot - margin) / encoded_absolute_dot
        if not math.isfinite(agreement):
            raise FloatingPointError("projection agreement is non-finite")
        return min(1.0, max(0.0, agreement))

    def _ensure_group_state(self) -> None:
        super()._ensure_group_state()
        self._SCALE_POLICY._projection_seen(self.param_groups[0])

    def _make_direction(
        self,
        raw: list[tuple[Tensor, Tensor, dict]],
        base_loss_tensor: Tensor,
        *,
        previous_step: int,
    ) -> _Direction:
        direction = super()._make_direction(
            raw,
            base_loss_tensor,
            previous_step=previous_step,
        )
        direction.omega = self._projection_agreement(direction)
        if direction.projection_active:
            direction.omega = 0.0
        return direction

    def _commit_direction(
        self, direction: _Direction, *, step_number: int
    ) -> None:
        group = self.param_groups[0]
        previous_seen = self._SCALE_POLICY._projection_seen(group)
        if type(direction.projection_active) is not bool:
            raise ValueError("direction projection marker must be a bool")
        super()._commit_direction(direction, step_number=step_number)
        group[self._SCALE_POLICY._PROJECTION_SEEN_KEY] = (
            previous_seen or direction.projection_active
        )

    @property
    def last_info(self) -> dict[str, Any] | None:
        result = super().last_info
        if result is not None:
            result["projection_seen"] = self._SCALE_POLICY._projection_seen(
                self.param_groups[0]
            )
        return result


class YamAdam(_ReleaseYamAdam):
    """Parameter-only optimizer with automatic memory and step scale.

    Uses neutral coordinate memory and global history projection, with a
    self-calibrated capped-max radius, a geometric support release, and an
    initial-backtrack bracket floor.  See :meth:`step` for the closure contract.
    """

    _PREFIX = "yamadam_initial_backtrack_bracket_floor_"
    _COEFFICIENT = f"{_PREFIX}coefficient"
    _INITIAL_DIRECTION_NORM = f"{_PREFIX}initial_direction_norm"
    _LAST_DIRECTION_NORM = f"{_PREFIX}last_direction_norm"
    _LAST_RELEASE_RADIUS = f"{_PREFIX}last_release_radius"
    _LAST_FLOOR_RADIUS = f"{_PREFIX}last_floor_radius"
    _LAST_APPLIED_RADIUS = f"{_PREFIX}last_applied_radius"
    _LAST_LIFTED = f"{_PREFIX}last_lifted"
    _LIFT_COUNT = f"{_PREFIX}lift_count"
    _STATE_KEYS = (
        _COEFFICIENT,
        _INITIAL_DIRECTION_NORM,
        _LAST_DIRECTION_NORM,
        _LAST_RELEASE_RADIUS,
        _LAST_FLOOR_RADIUS,
        _LAST_APPLIED_RADIUS,
        _LAST_LIFTED,
        _LIFT_COUNT,
    )
    _PENDING_BACKTRACKS = (
        "_yamadam_initial_backtrack_bracket_floor_pending_backtracks"
    )

    @staticmethod
    def _positive_finite(value: object, *, name: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a numeric scalar")
        result = float(value)
        if not math.isfinite(result) or result <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
        return result

    @staticmethod
    def _checked_direction_norm(direction: _Direction) -> float:
        norm = float(direction.direction_norm)
        if not math.isfinite(norm) or norm <= 0.0:
            raise FloatingPointError(
                "initial-backtrack bracket floor received an invalid "
                "direction norm"
            )
        return norm

    def _active(self) -> bool:
        return self._COEFFICIENT in self.param_groups[0]

    def _ensure_group_state(self) -> None:
        super()._ensure_group_state()
        group = self.param_groups[0]
        present = tuple(key in group for key in self._STATE_KEYS)
        if not any(present):
            if (
                bool(group["yamadam_initialized"])
                and int(group["yamadam_initial_backtracks"]) > 0
            ):
                raise ValueError(
                    "an active initial-backtrack-bracket-floor checkpoint "
                    "lacks its coefficient state"
                )
            return
        if not all(present):
            raise ValueError(
                "initial-backtrack-bracket-floor checkpoint state is incomplete"
            )
        if (
            not bool(group["yamadam_initialized"])
            or int(group["yamadam_initial_backtracks"]) <= 0
        ):
            raise ValueError(
                "initial-backtrack-bracket-floor state requires initial "
                "backtracking"
            )

        coefficient = self._positive_finite(
            group[self._COEFFICIENT],
            name="initial-backtrack bracket coefficient",
        )
        initial_norm = self._positive_finite(
            group[self._INITIAL_DIRECTION_NORM],
            name="initial-backtrack bracket initial direction norm",
        )
        calibration = self._positive_finite(
            group["yamadam_radius"],
            name="initial-backtrack bracket calibration radius",
        )
        if coefficient != (2.0 * calibration) / initial_norm:
            raise ValueError(
                "initial-backtrack bracket coefficient differs from "
                "2*C/||d_1||"
            )
        for key in (
            self._LAST_DIRECTION_NORM,
            self._LAST_RELEASE_RADIUS,
            self._LAST_FLOOR_RADIUS,
            self._LAST_APPLIED_RADIUS,
        ):
            self._positive_finite(group[key], name=key)
        if type(group[self._LAST_LIFTED]) is not bool:
            raise ValueError(
                "initial-backtrack bracket lifted marker must be a bool"
            )
        lift_count = group[self._LIFT_COUNT]
        if type(lift_count) is not int or not 0 <= lift_count <= int(
            group["yamadam_steps"]
        ):
            raise ValueError(
                "initial-backtrack bracket lift count is invalid"
            )

    def _initial_armijo(
        self, direction: _Direction, closure: BatchClosure
    ) -> tuple[float, Tensor, float, int, int, int, float, bool, list[Tensor]]:
        result = super()._initial_armijo(direction, closure)
        backtracks = result[5]
        if type(backtracks) is not int or backtracks < 0:
            raise RuntimeError("initial Armijo returned an invalid backtrack count")
        setattr(self, self._PENDING_BACKTRACKS, backtracks)
        return result

    def _initialize_scale_proposal(
        self,
        direction: _Direction,
        *,
        calibration_radius: float,
    ) -> ScaleProposal:
        proposal = super()._initialize_scale_proposal(
            direction,
            calibration_radius=calibration_radius,
        )
        backtracks = getattr(self, self._PENDING_BACKTRACKS, None)
        if type(backtracks) is not int or backtracks < 0:
            raise RuntimeError(
                "initial-backtrack bracket floor did not observe the Armijo "
                "result"
            )
        if backtracks == 0:
            return proposal

        norm = self._checked_direction_norm(direction)
        coefficient = (2.0 * calibration_radius) / norm
        if not math.isfinite(coefficient) or coefficient <= 0.0:
            raise FloatingPointError(
                "initial-backtrack bracket coefficient is invalid"
            )
        updates = dict(proposal.updates)
        updates.update(
            {
                self._COEFFICIENT: coefficient,
                self._INITIAL_DIRECTION_NORM: norm,
                self._LAST_DIRECTION_NORM: norm,
                self._LAST_RELEASE_RADIUS: proposal.radius,
                self._LAST_FLOOR_RADIUS: calibration_radius,
                self._LAST_APPLIED_RADIUS: proposal.radius,
                self._LAST_LIFTED: False,
                self._LIFT_COUNT: 0,
            }
        )
        return replace(proposal, updates=updates)

    def _steady_scale_proposal(self, direction: _Direction) -> ScaleProposal:
        release = super()._steady_scale_proposal(direction)
        if not self._active():
            return release

        group = self.param_groups[0]
        norm = self._checked_direction_norm(direction)
        coefficient = self._positive_finite(
            group[self._COEFFICIENT],
            name="initial-backtrack bracket coefficient",
        )
        calibration = self._positive_finite(
            group["yamadam_radius"],
            name="initial-backtrack bracket calibration radius",
        )
        raw_floor = coefficient * norm
        if not math.isfinite(raw_floor) or raw_floor <= 0.0:
            raise FloatingPointError(
                "initial-backtrack bracket coefficient floor is invalid"
            )
        floor = min(calibration, raw_floor)
        radius = max(release.radius, floor)
        if (
            not math.isfinite(release.radius)
            or release.radius <= 0.0
            or not math.isfinite(floor)
            or floor <= 0.0
            or not math.isfinite(radius)
            or radius < release.radius
            or radius < floor
            or radius > calibration
        ):
            raise FloatingPointError(
                "initial-backtrack bracket applied radius is invalid"
            )
        lifted = radius > release.radius
        previous_lifts = group[self._LIFT_COUNT]
        if type(previous_lifts) is not int or previous_lifts < 0:
            raise ValueError(
                "initial-backtrack bracket lift count is invalid"
            )
        updates = dict(release.updates)
        updates.update(
            {
                self._COEFFICIENT: coefficient,
                self._INITIAL_DIRECTION_NORM: float(
                    group[self._INITIAL_DIRECTION_NORM]
                ),
                self._LAST_DIRECTION_NORM: norm,
                self._LAST_RELEASE_RADIUS: release.radius,
                self._LAST_FLOOR_RADIUS: floor,
                self._LAST_APPLIED_RADIUS: radius,
                self._LAST_LIFTED: lifted,
                self._LIFT_COUNT: previous_lifts + int(lifted),
            }
        )
        return replace(release, radius=radius, updates=updates)

    def _floor_diagnostics(self) -> dict[str, bool | float | int | str]:
        group = self.param_groups[0]
        limiter = str(group["yamadam_initial_radius_limiter"])
        rejected_ray = (
            "production_direction"
            if limiter == "production_recheck"
            else "auxiliary_probe"
        )
        return {
            "initial_backtrack_bracket_floor_rule": (
                "max(release_radius,min(C,eta_bracket*direction_norm))"
            ),
            "initial_backtrack_bracket_floor_active": True,
            "initial_backtrack_bracket_floor_coefficient": float(
                group[self._COEFFICIENT]
            ),
            "initial_backtrack_bracket_floor_rejected_radius": (
                2.0 * float(group["yamadam_radius"])
            ),
            "initial_backtrack_bracket_floor_rejected_ray": rejected_ray,
            "initial_backtrack_bracket_floor_coefficient_is_extrapolated": (
                rejected_ray == "auxiliary_probe"
            ),
            "initial_backtrack_bracket_floor_initial_direction_norm": float(
                group[self._INITIAL_DIRECTION_NORM]
            ),
            "initial_backtrack_bracket_floor_last_direction_norm": float(
                group[self._LAST_DIRECTION_NORM]
            ),
            "initial_backtrack_bracket_floor_last_release_radius": float(
                group[self._LAST_RELEASE_RADIUS]
            ),
            "initial_backtrack_bracket_floor_last_floor_radius": float(
                group[self._LAST_FLOOR_RADIUS]
            ),
            "initial_backtrack_bracket_floor_last_applied_radius": float(
                group[self._LAST_APPLIED_RADIUS]
            ),
            "initial_backtrack_bracket_floor_last_lifted": bool(
                group[self._LAST_LIFTED]
            ),
            "initial_backtrack_bracket_floor_lift_count": int(
                group[self._LIFT_COUNT]
            ),
            "initial_backtrack_bracket_floor_extra_forwards": 0,
        }

    @property
    def diagnostics(self) -> dict[str, bool | float | int | str]:
        result = dict(super().diagnostics)
        if self._active():
            result.update(self._floor_diagnostics())
        return result

    @property
    def last_info(self) -> dict[str, Any] | None:
        result = super().last_info
        if result is not None and self._active():
            result["initial_backtrack_rejected_bracket_floor"] = (
                self._floor_diagnostics()
            )
        return result

    def step(self, closure: BatchClosure | None = None) -> Tensor:
        try:
            return super().step(closure)
        finally:
            self.__dict__.pop(self._PENDING_BACKTRACKS, None)


__all__ = ["BatchClosure", "ReplayableBatchClosure", "YamAdam"]
