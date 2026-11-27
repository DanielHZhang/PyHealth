"""NHANES 2003-2006 accelerometry data and public-use mortality linkage."""

import json
from collections.abc import Generator, Iterator, Sequence
from contextlib import contextmanager
from operator import index
from pathlib import Path
from shutil import copyfileobj
from tempfile import TemporaryDirectory
from typing import SupportsIndex, TypedDict, cast
from zipfile import ZipFile

import narwhals as nw
import pandas as pd

from .base_dataset import BaseDataset

CYCLES = {"2003-2004": "C", "2005-2006": "D"}
ACTIVITY_COLUMNS = (
    "SEQN",
    "PAXN",
    "PAXDAY",
    "PAXHOUR",
    "PAXMINUT",
    "PAXINTEN",
    "PAXSTAT",
    "PAXCAL",
)
EXAM_COLUMNS = {
    "BMX": ("BMXBMI", "BMXWAIST"),
    "BPX": tuple(f"{kind}{i}" for kind in ("BPXSY", "BPXDI") for i in range(1, 5)),
}
PARTICIPANT_INDEX = "participants.parquet"


class CycleSummary(TypedDict):
    demographics: int
    recordings: int
    recordings_with_demographics: int
    recordings_with_mortality_record: int
    minute_rows: int


class NHANESDataset(BaseDataset):
    """Load participant metadata and recording paths without filtering a cohort.

    Use :meth:`NHANESDataset.prepare` to convert local CDC files first. Each
    ``participants`` event has an ``activity_file`` referencing raw minute
    data. Linkage and exam fields are nullable. Changing recordings requires a new
    directory, as cached samples cannot reference an older version.

    Args:
        root: Prepared directory containing the participant index and recordings.
        dataset_name: Optional PyHealth dataset name.
        config_path: Optional YAML configuration. Defaults to ``nhanes.yaml``.
        **kwargs: Additional arguments forwarded to :class:`BaseDataset`.

    Examples:
        >>> from pyhealth.datasets import NHANESDataset
        >>> prepared = NHANESDataset.prepare("raw", "prepared")
        >>> dataset = NHANESDataset(root=prepared)
        >>> dataset.stats()
    """

    def __init__(
        self,
        root: str | Path,
        dataset_name: str | None = None,
        config_path: str | Path | None = None,
        **kwargs,
    ) -> None:
        root = Path(root).expanduser().resolve()
        if config_path is None:
            config_path = Path(__file__).parent / "configs" / "nhanes.yaml"
        super().__init__(
            root=str(root),
            tables=["participants"],
            dataset_name=dataset_name or "nhanes",
            config_path=str(config_path),
            **kwargs,
        )
        if self.config is None:
            raise ValueError("NHANES requires a dataset config")
        table = self.config.tables.get("participants")
        if table is None:
            raise ValueError("NHANES config must define a participants table")
        index = root / table.file_path
        if not index.exists():
            raise FileNotFoundError(
                f"Missing NHANES participant index: {index}. "
                "Run NHANESDataset.prepare first."
            )

    def preprocess_participants(self, frame: nw.LazyFrame) -> nw.LazyFrame:
        """Hook for the ``participants`` table that resolves paths relative to the
        current dataset root.
        """
        root_prefix = f"{Path(self.root).as_posix()}/"
        relative_path = nw.col("activity_file")
        absolute_path = nw.lit(root_prefix) + relative_path
        return frame.with_columns(absolute_path.alias("activity_file"))

    @staticmethod
    def prepare(
        raw_root: str | Path,
        output_dir: str | Path,
        *,
        cycles: Sequence[str] = ("2003-2004", "2005-2006"),
        include_exams: bool = False,
        chunk_size: int = 250_000,
    ) -> Path:
        """Convert local CDC files into a new directory for :class:`NHANESDataset`.

        Download the CDC files locally and arrange them under ``raw_root``:

        .. code-block:: text
            raw/
              2003-2004/DEMO_C.XPT
              2003-2004/PAXRAW_C.zip
              2005-2006/DEMO_D.XPT
              2005-2006/PAXRAW_D.zip
              mortality/NHANES_2003_2004_MORT_2019_PUBLIC.dat
              mortality/NHANES_2005_2006_MORT_2019_PUBLIC.dat

        Each survey file can be a SAS Transport file (``.xpt``), a ZIP archive
        containing one ``.xpt`` file, or a CSV with the original column names.
        ``PAXRAW`` must be ordered by ``SEQN``. For BMI, waist circumference,
        and blood pressure, place the matching ``BMX`` and ``BPX`` files in each
        selected cycle directory (``_C`` for 2003-2004, ``_D`` for 2005-2006)
        and pass ``include_exams=True``.

        The participant index stores source mortality fields and relative paths
        to minute recordings. It includes every participant with a recording,
        even if demographics or mortality linkage are missing. No cohort or
        wear filters are applied. Conversion counts are written to
        ``preparation_summary.json``. Partially converted data are not published,
        and existing output is not replaced.

        Args:
            raw_root: Directory containing local CDC survey and mortality files.
            output_dir: New directory for the participant index and recordings.
            cycles: Survey cycles to convert. Defaults to both supported cycles.
            include_exams: Whether to include body measurements and blood pressure.
            chunk_size: Maximum number of activity rows read per chunk.

        Returns:
            The resolved output directory to pass as ``NHANESDataset(root=...)``.

        Raises:
            FileExistsError: If the output directory already exists.
            FileNotFoundError: If a required source file is missing.
            ValueError: If the preparation options or source data are invalid.
        """

        cycles = tuple(cycles)
        if not cycles or len(set(cycles)) != len(cycles) or set(cycles) - CYCLES.keys():
            raise ValueError(
                f"cycles must be a nonempty selection from {tuple(CYCLES)}"
            )
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")

        raw_root = Path(raw_root).expanduser().resolve()
        output_dir = Path(output_dir).expanduser().resolve()
        if output_dir.exists():
            raise FileExistsError(f"Prepared output already exists: {output_dir}")
        output_dir.parent.mkdir(parents=True, exist_ok=True)

        with TemporaryDirectory(dir=output_dir.parent, prefix="nhanes-") as tmp:
            stage = Path(tmp) / "prepared"
            (stage / "activity").mkdir(parents=True)

            parts: list[pd.DataFrame] = []
            summaries: dict[str, CycleSummary] = {}
            seen_demo: set[int] = set()
            seen_activity: set[int] = set()

            for cycle in cycles:
                folder = raw_root / cycle
                suffix = CYCLES[cycle]

                demo = NHANESDataset.read_metadata(
                    folder,
                    f"DEMO_{suffix}",
                    ("RIDAGEYR", "RIAGENDR", "INDHHINC"),
                    temp_parent=output_dir.parent,
                    optional=("INDHHINC",),
                )
                demo_ids = set(demo["SEQN"])
                if seen_demo & demo_ids:
                    raise ValueError("Duplicate demographic SEQN across NHANES cycles")
                seen_demo.update(demo_ids)

                mortality_path = (
                    raw_root
                    / "mortality"
                    / f"NHANES_{cycle.replace('-', '_')}_MORT_2019_PUBLIC.dat"
                )
                columns = ("SEQN", "ELIGSTAT", "MORTSTAT", "PERMTH_INT")
                raw = pd.read_fwf(
                    mortality_path,
                    colspecs=[(0, 6), (14, 15), (15, 16), (42, 45)],
                    names=list(columns),
                    na_values=["."],
                )

                mortality = NHANESDataset.numeric_columns(
                    raw, mortality_path, columns
                ).astype(dict.fromkeys(columns[1:], "Int64"))
                if mortality["SEQN"].duplicated().any():
                    raise ValueError(f"Duplicate SEQN in {mortality_path}")
                if mortality["PERMTH_INT"].lt(0).any():
                    raise ValueError(
                        f"Negative mortality follow-up months in {mortality_path}"
                    )

                participant_fields = demo.merge(
                    mortality, how="outer", on="SEQN", validate="one_to_one"
                )

                for table, columns in EXAM_COLUMNS.items():
                    if include_exams:
                        exam = NHANESDataset.read_metadata(
                            folder,
                            f"{table}_{suffix}",
                            columns,
                            temp_parent=output_dir.parent,
                        )
                        participant_fields = participant_fields.merge(
                            exam, how="outer", on="SEQN", validate="one_to_one"
                        )
                    else:
                        for column in columns:
                            participant_fields[column] = float("nan")

                recordings: list[tuple[int, str]] = []
                minute_rows = 0

                with NHANESDataset.survey_file(
                    folder, f"PAXRAW_{suffix}", temp_parent=output_dir.parent
                ) as path:
                    for block in NHANESDataset.activity_blocks(path, chunk_size):
                        for key, records in block.groupby("SEQN", sort=False):
                            if not isinstance(key, SupportsIndex):
                                raise TypeError(f"Expected integer SEQN, got {key!r}")
                            seqn = int(key)
                            if seqn in seen_activity:
                                raise ValueError(
                                    "Duplicate activity SEQN across NHANES cycles: "
                                    f"{seqn}"
                                )

                            seen_activity.add(seqn)
                            records = records.sort_values("PAXN")
                            minutes = records["PAXN"]

                            if (
                                not minutes.between(1, 10_080).all()
                                or (minutes % 1 != 0).any()
                                or minutes.duplicated().any()
                            ):
                                raise ValueError(
                                    "Invalid or duplicate PAXN for "
                                    f"SEQN {seqn} in {cycle}"
                                )

                            relative_path = f"activity/{seqn}.parquet"
                            records.assign(PAXN=minutes.astype("int32")).to_parquet(
                                stage / relative_path, index=False
                            )
                            recordings.append((seqn, relative_path))
                            minute_rows += len(records)

                index = pd.DataFrame(
                    recordings, columns=["SEQN", "activity_file"]
                ).astype({"SEQN": "int64", "activity_file": "string"})

                participants = index.merge(
                    participant_fields, how="left", on="SEQN", validate="one_to_one"
                )
                participants["cycle"] = cycle

                parts.append(participants)
                summaries[cycle] = {
                    "demographics": len(demo),
                    "recordings": len(index),
                    # Cast to int needed for json.dumps to serialize
                    "recordings_with_demographics": int(
                        index["SEQN"].isin(demo_ids).sum()
                    ),
                    "recordings_with_mortality_record": int(
                        index["SEQN"].isin(mortality["SEQN"]).sum()
                    ),
                    "minute_rows": minute_rows,
                }

            participants = pd.concat(parts, ignore_index=True).sort_values("SEQN")
            if participants.empty:
                raise ValueError("No NHANES activity recordings found")

            participants.columns = participants.columns.str.lower()
            participants.to_parquet(stage / PARTICIPANT_INDEX, index=False)

            summary = {
                "participants": len(participants),
                "include_exams": include_exams,
                "cycles": summaries,
            }
            (stage / "preparation_summary.json").write_text(
                json.dumps(summary, indent=2) + "\n", encoding="utf-8"
            )
            stage.rename(output_dir)

        return output_dir

    @staticmethod
    @contextmanager
    def survey_file(folder: Path, stem: str, temp_parent: Path) -> Generator:
        """Locate a survey file and extract it temporarily if it is zipped.

        Args:
            folder: Cycle directory to search for the survey file.
            stem: CDC filename without its extension.
            temp_parent: Directory in which to extract a zipped XPORT file.

        Yields:
            A readable CSV or XPORT path that is valid until the context exits.
        """

        names = (
            {path.name.lower(): path for path in folder.iterdir()}
            if folder.is_dir()
            else {}
        )

        for extension in (".xpt", ".xpt.zip", ".zip", ".csv"):
            path = names.get(stem.lower() + extension)
            if path is None or not path.is_file():
                continue
            if path.suffix.lower() != ".zip":
                yield path
                return

            with (
                ZipFile(path) as archive,
                TemporaryDirectory(dir=temp_parent, prefix="nhanes-xport-") as tmp,
            ):
                items = [item for item in archive.infolist() if not item.is_dir()]
                if len(items) != 1 or Path(items[0].filename).suffix.lower() != ".xpt":
                    raise ValueError(f"Expected one XPORT file in {path}")

                extracted = Path(tmp) / f"{stem}.xpt"
                with archive.open(items[0]) as source, extracted.open("wb") as dest:
                    copyfileobj(source, dest)
                yield extracted

            return

        raise FileNotFoundError(f"Missing NHANES survey file {stem} in {folder}")

    @staticmethod
    def numeric_columns(
        frame: pd.DataFrame,
        path: Path,
        columns: tuple[str, ...],
        optional: tuple[str, ...] = (),
    ) -> pd.DataFrame:
        """Select numeric source fields and validate the participant ID.

        Args:
            frame: Source data, including the ``SEQN`` column.
            path: Source path used to identify the format and report errors.
            columns: Columns to retain, including ``SEQN``.
            optional: Selected columns that may be missing from the source.

        Returns:
            Numeric data with integer participant IDs and nulls for missing fields.
        """

        missing = set(columns) - set(optional) - set(frame.columns)
        if missing:
            raise ValueError(f"Missing columns in {path}: {sorted(missing)}")

        frame = frame.reindex(columns=list(columns))
        frame = frame.apply(pd.to_numeric, errors="raise")
        if path.suffix.lower() == ".xpt":
            # Pandas can decode an IBM-format zero as a tiny nonzero float.
            frame = frame.mask(frame.abs().lt(1e-10), 0)

        ids = frame["SEQN"]
        if not ids.between(1, 999_999).all() or (ids % 1 != 0).any():
            raise ValueError(
                f"Invalid SEQN in {path}; expected positive integer participant IDs"
            )

        frame["SEQN"] = ids.astype("int64")
        return frame

    @staticmethod
    def read_metadata(
        folder: Path,
        file_stem: str,
        columns: tuple[str, ...],
        temp_parent: Path,
        optional: tuple[str, ...] = (),
    ) -> pd.DataFrame:
        """Read a demographics or exam table for a survey cycle.

        Args:
            folder: Cycle directory containing the survey file.
            stem: CDC filename without its extension.
            columns: Metadata columns to read in addition to ``SEQN``.
            temp_parent: Directory in which to extract a zipped XPORT file.
            optional: Metadata columns that may be missing from the source.

        Returns:
            Numeric source fields with at most one row per participant.
        """
        with NHANESDataset.survey_file(folder, file_stem, temp_parent) as path:
            df_raw = (
                pd.read_csv(path)
                if path.suffix.lower() == ".csv"
                else pd.read_sas(path, format="xport", encoding="utf-8")
            )
            frame = NHANESDataset.numeric_columns(
                df_raw, path, ("SEQN", *columns), optional
            )
            if frame["SEQN"].duplicated().any():
                raise ValueError(f"Duplicate SEQN in {path}")
            return frame

    @staticmethod
    def activity_blocks(path: Path, chunk_size: int) -> Iterator[pd.DataFrame]:
        """Stream activity records without splitting a participant across blocks.

        Args:
            path: CSV or XPORT activity file ordered by participant ID.
            chunk_size: Maximum number of rows read at a time.

        Yields:
            Blocks containing one or more complete participant recordings.
        """

        # Preserve the last participant until its entire recording is read.
        tail = None
        last_id: int | None = None
        reader = (
            pd.read_csv(path, chunksize=chunk_size)
            if path.suffix.lower() == ".csv"
            else pd.read_sas(
                path, format="xport", encoding="utf-8", chunksize=chunk_size
            )
        )

        with reader:
            for df_raw in reader:
                chunk = NHANESDataset.numeric_columns(df_raw, path, ACTIVITY_COLUMNS)
                if chunk.empty:
                    continue

                ids = chunk["SEQN"]
                # numeric_columns validates SEQN and converts it to int64
                first_id = index(cast(SupportsIndex, ids.iat[0]))
                if not ids.is_monotonic_increasing or (
                    last_id is not None and first_id < last_id
                ):
                    raise ValueError(f"NHANES activity must be ordered by SEQN: {path}")

                last_id = index(cast(SupportsIndex, ids.iat[-1]))
                if tail is not None:
                    chunk = pd.concat([tail, chunk], ignore_index=True)

                boundary = chunk["SEQN"].searchsorted(last_id)
                if boundary:
                    yield chunk.iloc[:boundary]

                tail = chunk.iloc[boundary:].copy()

        if tail is not None:
            yield tail
