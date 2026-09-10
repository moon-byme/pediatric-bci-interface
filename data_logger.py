import csv
from datetime import datetime
from pathlib import Path


class TrialLogger:
    """Stores one row for each completed interaction target."""

    def __init__(self, data_directory=None):
        if data_directory is None:
            data_directory = (
                Path(__file__).parent
                / "data"
            )

        self.data_directory = Path(data_directory)
        self.data_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        session_id = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        self.file_path = (
            self.data_directory
            / f"interaction_session_{session_id}.csv"
        )

        with self.file_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "trial",
                    "arm",
                    "target_x",
                    "target_y",
                    "hold_time_seconds",
                    "completed_at",
                ]
            )

    def log_success(
        self,
        trial,
        arm,
        target_x,
        target_y,
        hold_time_seconds,
    ):
        with self.file_path.open(
            "a",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    trial,
                    arm,
                    target_x,
                    target_y,
                    round(hold_time_seconds, 3),
                    datetime.now().isoformat(
                        timespec="seconds"
                    ),
                ]
            )
