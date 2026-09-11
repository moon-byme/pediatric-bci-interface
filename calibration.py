import statistics
import time


CALIBRATION_STAGES = [
    (
        "center",
        "Place your wrist in a comfortable neutral position",
    ),
    (
        "left",
        "Move your wrist comfortably toward the LEFT side of the screen",
    ),
    (
        "right",
        "Move your wrist comfortably toward the RIGHT side of the screen",
    ),
    (
        "up",
        "Move your wrist comfortably UP",
    ),
    (
        "down",
        "Move your wrist comfortably DOWN",
    ),
]


class EMAFilter:
    """Simple exponential moving average for 2D interaction coordinates."""

    def __init__(self, alpha=0.25):
        self.alpha = alpha
        self.x = None
        self.y = None

    def reset(self):
        self.x = None
        self.y = None

    def update(self, x, y):
        if self.x is None or self.y is None:
            self.x = float(x)
            self.y = float(y)
        else:
            self.x = (
                self.alpha * float(x)
                + (1.0 - self.alpha) * self.x
            )
            self.y = (
                self.alpha * float(y)
                + (1.0 - self.alpha) * self.y
            )

        return int(self.x), int(self.y)


class ArmWorkspaceCalibrator:
    """
    Estimate one arm's comfortable 2D interaction workspace.

    This is an interface calibration, not a clinical measurement of joint ROM.
    """

    def __init__(
        self,
        capture_duration=0.7,
        min_samples=8,
        min_visibility=0.55,
    ):
        self.capture_duration = capture_duration
        self.min_samples = min_samples
        self.min_visibility = min_visibility
        self.reset()

    def reset(self):
        self.stage_index = 0
        self.points = {}
        self.capturing = False
        self.capture_start = None
        self.samples = []
        self.last_status = "Press C to capture this position"

    @property
    def complete(self):
        return self.stage_index >= len(CALIBRATION_STAGES)

    @property
    def stage_name(self):
        if self.complete:
            return "complete"
        return CALIBRATION_STAGES[self.stage_index][0]

    @property
    def prompt(self):
        if self.complete:
            return "Calibration completed"
        return CALIBRATION_STAGES[self.stage_index][1]

    @property
    def stage_number(self):
        return min(
            self.stage_index + 1,
            len(CALIBRATION_STAGES),
        )

    @property
    def stage_count(self):
        return len(CALIBRATION_STAGES)

    def start_capture(self):
        if self.complete or self.capturing:
            return

        self.capturing = True
        self.capture_start = time.perf_counter()
        self.samples = []
        self.last_status = "Hold the selected arm steady..."

    def capture_progress(self):
        if not self.capturing or self.capture_start is None:
            return 0.0

        elapsed = time.perf_counter() - self.capture_start
        return min(elapsed / self.capture_duration, 1.0)

    def update(self, wrist, arm_reliable=True):
        """
        Add one sample only when the selected arm is considered reliable.

        Returns:
            None, "captured", "retry" or "complete".
        """
        if not self.capturing:
            return None

        if (
            arm_reliable
            and wrist is not None
            and wrist.get("visibility", 0.0) >= self.min_visibility
        ):
            self.samples.append(
                (
                    wrist["pixel_x"],
                    wrist["pixel_y"],
                )
            )

        elapsed = time.perf_counter() - self.capture_start

        if elapsed < self.capture_duration:
            return None

        self.capturing = False
        self.capture_start = None

        if len(self.samples) < self.min_samples:
            self.samples = []
            self.last_status = (
                "Selected arm was not visible/stable enough. Press C to retry"
            )
            return "retry"

        xs = [sample[0] for sample in self.samples]
        ys = [sample[1] for sample in self.samples]

        point = {
            "x": int(statistics.median(xs)),
            "y": int(statistics.median(ys)),
            "sample_count": len(self.samples),
        }

        current_stage = CALIBRATION_STAGES[
            self.stage_index
        ][0]

        self.points[current_stage] = point
        self.samples = []
        self.stage_index += 1

        if self.complete:
            self.last_status = "Calibration completed for this arm"
            return "complete"

        self.last_status = "Position captured. Press C for the next position"
        return "captured"

    def get_workspace(self):
        if not self.complete:
            return None

        left_x = self.points["left"]["x"]
        right_x = self.points["right"]["x"]
        up_y = self.points["up"]["y"]
        down_y = self.points["down"]["y"]

        min_x = min(left_x, right_x)
        max_x = max(left_x, right_x)
        min_y = min(up_y, down_y)
        max_y = max(up_y, down_y)

        return {
            "min_x": min_x,
            "max_x": max_x,
            "min_y": min_y,
            "max_y": max_y,
            "width": max_x - min_x,
            "height": max_y - min_y,
            "center_x": self.points["center"]["x"],
            "center_y": self.points["center"]["y"],
        }

    def map_target(
        self,
        normalized_x,
        normalized_y,
        safe_margin=0.08,
    ):
        """Map 0..1 target coordinates into this arm's workspace."""
        workspace = self.get_workspace()

        if workspace is None:
            return None

        usable_ratio = 1.0 - 2.0 * safe_margin

        nx = safe_margin + normalized_x * usable_ratio
        ny = safe_margin + normalized_y * usable_ratio

        x = workspace["min_x"] + nx * workspace["width"]
        y = workspace["min_y"] + ny * workspace["height"]

        return int(x), int(y)


class BilateralWorkspaceManager:
    """Keep independent calibrations for the right and left arms."""

    def __init__(
        self,
        capture_duration=0.7,
        min_samples=8,
        min_visibility=0.55,
    ):
        self.calibrators = {
            arm: ArmWorkspaceCalibrator(
                capture_duration=capture_duration,
                min_samples=min_samples,
                min_visibility=min_visibility,
            )
            for arm in ("right", "left")
        }

    def get(self, arm):
        return self.calibrators[arm]

    def is_complete(self, arm):
        return self.get(arm).complete

    def reset_arm(self, arm):
        self.get(arm).reset()

    def reset_all(self):
        for calibrator in self.calibrators.values():
            calibrator.reset()
