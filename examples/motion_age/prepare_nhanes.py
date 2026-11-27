"""Convert local CDC files for use with NHANESDataset."""

import argparse

from pyhealth.datasets.nhanes import CYCLES, NHANESDataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_root")
    parser.add_argument("output_dir")
    parser.add_argument("--cycles", nargs="+", choices=CYCLES, default=tuple(CYCLES))
    parser.add_argument("--include-exams", action="store_true")
    args = parser.parse_args()
    prepared_path = NHANESDataset.prepare(
        args.raw_root,
        args.output_dir,
        cycles=args.cycles,
        include_exams=args.include_exams,
    )
    print(
        f"Prepared NHANES at {prepared_path}. See preparation_summary.json for counts."
    )


if __name__ == "__main__":
    main()
