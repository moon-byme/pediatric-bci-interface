import csv
import json
from datetime import datetime
from pathlib import Path


class TrialLogger:
    """Store arm-specific calibration metadata and completed trials."""

    def __init__(self, data_directory=None):
        if data_directory is None:
            data_directory = Path(__file__).parent / "data"

        self.data_directory = Path(data_directory)
        self.data_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.session_id = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        self.file_path = (
            self.data_directory
            / f"interaction_session_{self.session_id}.csv"
        )

        self.calibration_path = (
            self.data_directory
            / f"calibration_{self.session_id}.json"
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
                    "arm_trial",
                    "target_index",
                    "target_x",
                    "target_y",
                    "hold_time_seconds",
                    "min_arm_visibility",
                    "mean_arm_visibility",
                    "pose_model",
                    "completed_at",
                ]
            )

        self._write_calibration_payload(
            {
                "session_id": self.session_id,
                "created_at": datetime.now().isoformat(
                    timespec="seconds"
                ),
                "note": (
                    "Independent comfortable interaction workspaces for each arm; "
                    "not clinical joint-ROM measurements."
                ),
                "arms": {},
            }
        )

    def _read_calibration_payload(self):
        if not self.calibration_path.exists():
            return {
                "session_id": self.session_id,
                "arms": {},
            }

        with self.calibration_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)

    def _write_calibration_payload(self, payload):
        with self.calibration_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                payload,
                file,
                indent=2,
                ensure_ascii=False,
            )

    def save_arm_calibration(
        self,
        arm,
        calibration_points,
        workspace,
        pose_model,
    ):
        payload = self._read_calibration_payload()
        payload.setdefault("arms", {})

        payload["arms"][arm] = {
            "saved_at": datetime.now().isoformat(
                timespec="seconds"
            ),
            "pose_model": pose_model,
            "calibration_points": calibration_points,
            "workspace": workspace,
        }

        self._write_calibration_payload(payload)

    def log_success(
        self,
        trial,
        arm,
        arm_trial,
        target_index,
        target_x,
        target_y,
        hold_time_seconds,
        min_arm_visibility,
        mean_arm_visibility,
        pose_model,
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
                    arm_trial,
                    target_index,
                    target_x,
                    target_y,
                    round(hold_time_seconds, 3),
                    round(min_arm_visibility, 3),
                    round(mean_arm_visibility, 3),
                    pose_model,
                    datetime.now().isoformat(
                        timespec="seconds"
                    ),
                ]
            )
