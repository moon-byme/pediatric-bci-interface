import csv
import json
from datetime import datetime
from pathlib import Path


class TrialLogger:
    """Store shoulder-relative calibration metadata and completed trials."""

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
                    "target_screen_x",
                    "target_screen_y",
                    "target_relative_x",
                    "target_relative_y",
                    "hold_time_seconds",
                    "arm_scale_px",
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
                "coordinate_system": (
                    "2D shoulder-relative wrist offsets normalized by "
                    "projected arm-chain length (shoulder-elbow + elbow-wrist)"
                ),
                "note": (
                    "Independent interaction workspaces for each arm; "
                    "engineering calibration only, not a clinical ROM measure."
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
        comfortable_workspace,
        usable_workspace,
        usable_ratio,
        pose_model,
    ):
        payload = self._read_calibration_payload()
        payload.setdefault("arms", {})

        payload["arms"][arm] = {
            "saved_at": datetime.now().isoformat(
                timespec="seconds"
            ),
            "pose_model": pose_model,
            "usable_ratio": usable_ratio,
            "calibration_points": calibration_points,
            "comfortable_workspace_relative": comfortable_workspace,
            "usable_workspace_relative": usable_workspace,
        }

        self._write_calibration_payload(payload)

    def log_success(
        self,
        trial,
        arm,
        arm_trial,
        target_index,
        target_screen_x,
        target_screen_y,
        target_relative_x,
        target_relative_y,
        hold_time_seconds,
        arm_scale_px,
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
                    target_screen_x,
                    target_screen_y,
                    round(target_relative_x, 6),
                    round(target_relative_y, 6),
                    round(hold_time_seconds, 3),
                    round(arm_scale_px, 3),
                    round(min_arm_visibility, 3),
                    round(mean_arm_visibility, 3),
                    pose_model,
                    datetime.now().isoformat(
                        timespec="seconds"
                    ),
                ]
            )
