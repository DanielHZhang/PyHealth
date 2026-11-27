"""NHANES task for predicting 60-month mortality from accelerometer data."""

from itertools import pairwise
from math import isfinite
from typing import Any, Literal, NamedTuple, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch

from pyhealth.data import Event, Patient

from .base_task import BaseTask


class ActivityEpochs(NamedTuple):
    intensity: npt.NDArray[np.float64]
    hour: npt.NDArray[np.intp]
    day: npt.NDArray[np.intp]
    wear_mask: npt.NDArray[np.bool_]


class NHANESMortality60Month(BaseTask):
    """Task for predicting 60-month mortality from accelerometer data.

    Works with :class:`~pyhealth.datasets.NHANESDataset`.

    Each sample is a fixed-length window of a participant's activity recording.
    All windows from the same participant share a binary label indicating
    whether the participant died within 60 months of the NHANES interview.
    Participants whose follow-up does not establish this outcome are excluded.

    Each five-minute epoch contains the mean activity during wear minutes,
    hour of day, day of week, and a binary wear mask. Windows with insufficient
    wear coverage are dropped. Within retained windows, epochs that do not
    meet the wear threshold are masked rather than removed, preserving the
    regular five-minute spacing.

    Args:
        min_age: Minimum age at interview. The original implementation uses 18.
        seq_len: Window length in five-minute epochs.
        stride: Window step in epochs. Defaults to non-overlapping windows.
        epoch_coverage: Minimum fraction of worn minutes for a valid epoch.
        window_coverage: Minimum fraction of valid epochs for a retained window.
        require_calibration: Exclude minutes from monitors not in calibration.
        features: ``"wearable"`` (F), ``"age"`` (FA), or ``"routine"`` (FRC).
    """

    task_name: str = "NHANESMortality60Month"

    def __init__(
        self,
        min_age: int = 18,
        seq_len: int = 1008,
        stride: int | None = None,
        epoch_coverage: float = 0.2,
        window_coverage: float = 0.3,
        require_calibration: bool = True,
        features: Literal["wearable", "age", "routine"] = "wearable",
    ) -> None:
        super().__init__()

        if min_age < 0:
            raise ValueError("min_age must be nonnegative")
        if seq_len < 1:
            raise ValueError("seq_len must be positive")
        if stride is not None and stride < 1:
            raise ValueError("stride must be positive")
        if not (0 < epoch_coverage <= 1 and 0 < window_coverage <= 1):
            raise ValueError("Coverage thresholds must be in (0, 1]")
        if features not in ("wearable", "age", "routine"):
            raise ValueError("features must be wearable, age, or routine")

        self.min_age = min_age
        self.seq_len = seq_len
        self.stride = seq_len if stride is None else stride
        self.epoch_coverage = epoch_coverage
        self.window_coverage = window_coverage
        self.require_calibration = require_calibration
        self.features = features
        self.output_schema = {"mortality_60m": "binary"}
        input_schema: dict[str, str | tuple[str, dict[str, torch.dtype]]] = {
            "intensity": "tensor",
            "hour": ("tensor", {"dtype": torch.long}),
            "day": ("tensor", {"dtype": torch.long}),
            "wear_mask": ("tensor", {"dtype": torch.bool}),
        }
        if features != "wearable":
            input_schema["static_numeric"] = "tensor"
        if features == "routine":
            input_schema["static_categorical"] = ("tensor", {"dtype": torch.long})
        # The dataset accepts (processor, options) tuples, but BaseTask's annotation does not.
        self.input_schema = input_schema  # pyright: ignore[reportAttributeAccessIssue]

    def label(self, event: Event) -> int | None:
        """Assign a 60-month mortality label from the NHANES interview date.

        Returns:
            1 if the participant died within 60 months
            0 if the participant survived beyond that point
            None if the participant is ineligible or their follow-up is insufficient.
        """

        age, months = event.ridageyr, event.permth_int
        if (
            age is None
            or not isfinite(age)
            or age < self.min_age
            or event.eligstat != 1
            or months is None
            or not isfinite(months)
            or months < 0
        ):
            return None
        if event.mortstat == 1:
            return int(months <= 60)
        if event.mortstat == 0 and months >= 60:
            return 0
        return None

    @staticmethod
    def not_worn_mask(
        counts: np.ndarray, valid: npt.NDArray[np.bool_]
    ) -> npt.NDArray[np.bool_]:
        """Flag the minutes when the accelerometer was likely not worn.

        Uses a Choi-style rule to identify sustained periods of little or no
        activity. Invalid minutes are not flagged and break these periods.
        """

        not_worn = np.zeros(counts.size, dtype=bool)
        breaks = np.concatenate(
            ([-1], np.flatnonzero(~valid | (counts > 100)), [counts.size]),
            dtype=np.intp,
        )

        for left, right in pairwise(map(int, breaks)):
            start = left + 1
            if right - start < 90:
                continue

            spikes = np.flatnonzero(counts[start:right]) + start
            while right - start >= 90:
                first = int(spikes.searchsorted(start))
                # End the candidate window just before the third spike, or before the
                # second if it immediately follows the first.
                limit = min(first + 2, len(spikes))
                if first + 1 < len(spikes) and spikes[first + 1] == spikes[first] + 1:
                    limit = first + 1

                stop = int(spikes[limit]) if limit < len(spikes) else right
                included = spikes[first:limit]
                qualifies = stop - start >= 90

                if qualifies and included.size > 0:
                    quiet_before = included[0] - start
                    quiet_after = stop - included[-1] - 1
                    spikes_separated = (
                        included.size == 1 or included[1] - included[0] > 30
                    )
                    qualifies = (
                        quiet_before >= 30 and quiet_after >= 30 and spikes_separated
                    )

                if qualifies:
                    not_worn[start:stop] = True
                    start = stop
                else:
                    # Moving the start closer to the first spike only shortens the
                    # window, so retry after that spike.
                    start = int(spikes[first]) + 1
        return not_worn

    def __call__(self, patient: Patient) -> list[dict[str, Any]]:
        events = cast(list[Event], patient.get_events("participants", return_df=False))
        if len(events) != 1:
            raise ValueError(
                f"NHANES participant {patient.patient_id} must have one participant event"
            )

        event = events[0]
        target = self.label(event)
        if target is None:
            return []

        minutes = pd.read_parquet(
            event.activity_file,
            columns=[
                "PAXN",
                "PAXDAY",
                "PAXHOUR",
                "PAXMINUT",
                "PAXINTEN",
                "PAXSTAT",
                "PAXCAL",
            ],
        )
        epochs = self.build_epochs(minutes, patient.patient_id)
        if epochs is None:
            return []

        metadata = {
            "patient_id": patient.patient_id,
            "mortality_60m": target,
            **self.static_features(event),
        }
        samples = []
        for start in range(0, len(epochs.wear_mask) - self.seq_len + 1, self.stride):
            window = slice(start, start + self.seq_len)
            if epochs.wear_mask[window].mean() >= self.window_coverage:
                samples.append(
                    {
                        **metadata,
                        "window_start": start,
                        "intensity": epochs.intensity[window]
                        .astype(np.float32)
                        .tolist(),
                        "hour": epochs.hour[window].tolist(),
                        "day": epochs.day[window].tolist(),
                        "wear_mask": epochs.wear_mask[window].tolist(),
                    }
                )
        return samples

    def build_epochs(
        self, df_minutes: pd.DataFrame, patient_id: str
    ) -> ActivityEpochs | None:
        """Aggregates a minute recording on a regular five-minute grid."""

        if df_minutes.empty:
            return None

        first = df_minutes.iloc[0]
        time_bounds = (("PAXDAY", 1, 7), ("PAXHOUR", 0, 23), ("PAXMINUT", 0, 59))
        for field, lower, upper in time_bounds:
            values = cast(pd.Series, df_minutes[field])
            if not values.between(lower, upper).all() or values.mod(1).ne(0).any():
                raise ValueError(f"NHANES participant {patient_id} has invalid {field}")

        # The NHANES loader writes PAXN as int32 in ascending order.
        last_minute = int(cast(np.int32, df_minutes["PAXN"].iat[-1]))
        # Keep missing PAXN entries in the grid, including the unobserved positions
        # in a partial final epoch.
        epoch_count = (last_minute + 4) // 5
        if epoch_count < self.seq_len:
            return None

        grid = df_minutes.set_index("PAXN").reindex(range(1, 5 * epoch_count + 1))
        counts = grid["PAXINTEN"].to_numpy(dtype=float, na_value=np.nan)
        valid = (grid["PAXSTAT"].eq(1) & grid["PAXINTEN"].between(0, 32767)).to_numpy(
            dtype=bool
        )
        if self.require_calibration:
            valid &= grid["PAXCAL"].eq(1).to_numpy(dtype=bool)

        worn = valid & ~self.not_worn_mask(counts, valid)
        wear_minutes = worn.reshape(-1, 5).sum(axis=1)
        intensity = np.where(worn, counts, 0).reshape(-1, 5).sum(axis=1) / np.maximum(
            wear_minutes, 1
        )
        wear_mask = wear_minutes / 5 >= self.epoch_coverage

        # Anchor the clock to the first observed timestamp, then use PAXN to locate each
        # epoch, including any gaps before the first observation.
        recording_start_minute = (
            int(first.PAXHOUR) * 60 + int(first.PAXMINUT) - (int(first.PAXN) - 1)
        )
        epoch_minutes = recording_start_minute + 5 * np.arange(
            epoch_count, dtype=np.intp
        )
        hour = epoch_minutes // 60 % 24
        day = (int(first.PAXDAY) - 1 + epoch_minutes // 1440) % 7
        return ActivityEpochs(intensity, hour, day, wear_mask)

    def static_features(self, event: Event) -> dict[str, list[float] | list[int]]:
        """Return participant features that stay the same across activity windows."""

        if self.features == "wearable":
            return {}

        numeric: list[float] = [event.ridageyr]
        if self.features == "routine":
            # Missing values stay missing until the training partition fits its
            # processors. Zero is reserved for missing category codes.
            numeric.extend(
                float("nan") if x is None else x for x in (event.bmxbmi, event.bmxwaist)
            )

            for field in ("bpxsy", "bpxdi"):
                readings = [
                    cast(float | None, event.attr_dict[f"{field}{i}"])
                    for i in range(1, 5)
                ]
                values = [
                    value for value in readings if value is not None and isfinite(value)
                ]
                numeric.append(float(np.mean(values)) if values else float("nan"))

            sex = event.riagendr if event.riagendr in (1, 2) else 0
            income = event.indhhinc
            return {
                "static_numeric": numeric,
                "static_categorical": [
                    int(sex),
                    int(income) if income is not None and 1 <= income <= 13 else 0,
                ],
            }

        return {"static_numeric": numeric}
