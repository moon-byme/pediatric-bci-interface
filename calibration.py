import math
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

DEFAULT_USABLE_RATIO = 0.85
MIN_PROJECTED_ARM_SCALE = 40.0


class EMAFilter:
    """Exponential moving average for a 2D signal."""

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

        return self.x, self.y


class ShoulderReferenceFilter:
    """Smooth the shoulder anchor and projected arm scale independently."""

    def __init__(self, alpha=0.20):
        self.alpha = alpha
        self.shoulder_x = None
        self.shoulder_y = None
        self.arm_scale = None

    def reset(self):
        self.shoulder_x = None
        self.shoulder_y = None
        self.arm_scale = None

    def update(self, shoulder_x, shoulder_y, arm_scale):
        if self.shoulder_x is None:
            self.shoulder_x = float(shoulder_x)
            self.shoulder_y = float(shoulder_y)
            self.arm_scale = float(arm_scale)
        else:
            self.shoulder_x = (
                self.alpha * float(shoulder_x)
                + (1.0 - self.alpha) * self.shoulder_x
            )
            self.shoulder_y = (
                self.alpha * float(shoulder_y)
                + (1.0 - self.alpha) * self.shoulder_y
            )
            self.arm_scale = (
                self.alpha * float(arm_scale)
                + (1.0 - self.alpha) * self.arm_scale
            )

        return {
            "shoulder_x": self.shoulder_x,
            "shoulder_y": self.shoulder_y,
            "arm_scale": self.arm_scale,
        }


def get_shoulder_relative_state(landmarks, arm):
    """
    Express the selected wrist relative to the selected shoulder.

    The 2D offset is normalized by the projected arm-chain length
    (shoulder->elbow + elbow->wrist). This makes interaction less dependent
    on the participant's absolute location in the camera frame and partially
    compensates for image scale.

    This is an engineering coordinate system for interaction, not a clinical
    biomechanical measurement.
    """

    shoulder = landmarks.get(f"{arm}_shoulder")
    elbow = landmarks.get(f"{arm}_elbow")
    wrist = landmarks.get(f"{arm}_wrist")

    if shoulder is None or elbow is None or wrist is None:
        return None

    shoulder_x = float(shoulder["pixel_x"])
    shoulder_y = float(shoulder["pixel_y"])
    elbow_x = float(elbow["pixel_x"])
    elbow_y = float(elbow["pixel_y"])
    wrist_x = float(wrist["pixel_x"])
    wrist_y = float(wrist["pixel_y"])

    upper_arm = math.hypot(
        elbow_x - shoulder_x,
        elbow_y - shoulder_y,
    )
    forearm = math.hypot(
        wrist_x - elbow_x,
        wrist_y - elbow_y,
    )

    arm_scale = upper_arm + forearm

    if arm_scale < MIN_PROJECTED_ARM_SCALE:
        return None

    relative_x = (
        wrist_x - shoulder_x
    ) / arm_scale
    relative_y = (
        wrist_y - shoulder_y
    ) / arm_scale

    return {
        "relative_x": relative_x,
        "relative_y": relative_y,
        "shoulder_x": shoulder_x,
        "shoulder_y": shoulder_y,
        "arm_scale": arm_scale,
        "wrist_x": wrist_x,
        "wrist_y": wrist_y,
    }


def relative_to_screen(
    relative_x,
    relative_y,
    reference,
):
    """Project one shoulder-relative point back into display coordinates."""

    if reference is None:
        return None

    x = (
        reference["shoulder_x"]
        + relative_x * reference["arm_scale"]
    )
    y = (
        reference["shoulder_y"]
        + relative_y * reference["arm_scale"]
    )

    return int(round(x)), int(round(y))


class ArmWorkspaceCalibrator:
    """
    Estimate one arm's comfortable shoulder-relative interaction workspace.

    Five positions are collected (neutral + four cardinal directions). The
    resulting workspace is stored in shoulder-relative units, not absolute
    screen pixels. A conservative usable workspace is then obtained by
    shrinking each directional extent toward the neutral point.

    This is an interface calibration, not a clinical measurement of joint ROM.
    """

    def __init__(
        self,
        capture_duration=0.7,
        min_samples=8,
        min_visibility=0.55,
        usable_ratio=DEFAULT_USABLE_RATIO,
    ):
        self.capture_duration = capture_duration
        self.min_samples = min_samples
        self.min_visibility = min_visibility
        self.usable_ratio = usable_ratio
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

    def update(self, relative_state, arm_reliable=True):
        """
        Add one sample only when the selected arm is considered reliable.

        Returns:
            None, "captured", "retry" or "complete".
        """
        if not self.capturing:
            return None

        if arm_reliable and relative_state is not None:
            self.samples.append(
                (
                    relative_state["relative_x"],
                    relative_state["relative_y"],
                    relative_state["shoulder_x"],
                    relative_state["shoulder_y"],
                    relative_state["arm_scale"],
                    relative_state["wrist_x"],
                    relative_state["wrist_y"],
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

        columns = list(zip(*self.samples))

        point = {
            "relative_x": float(statistics.median(columns[0])),
            "relative_y": float(statistics.median(columns[1])),
            "capture_shoulder_x": float(statistics.median(columns[2])),
            "capture_shoulder_y": float(statistics.median(columns[3])),
            "capture_arm_scale": float(statistics.median(columns[4])),
            "capture_wrist_x": float(statistics.median(columns[5])),
            "capture_wrist_y": float(statistics.median(columns[6])),
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
        """Return the full comfortable workspace in shoulder-relative units."""
        if not self.complete:
            return None

        center_x = self.points["center"]["relative_x"]
        center_y = self.points["center"]["relative_y"]

        min_x = min(
            self.points["left"]["relative_x"],
            center_x,
            self.points["right"]["relative_x"],
        )
        max_x = max(
            self.points["left"]["relative_x"],
            center_x,
            self.points["right"]["relative_x"],
        )
        min_y = min(
            self.points["up"]["relative_y"],
            center_y,
            self.points["down"]["relative_y"],
        )
        max_y = max(
            self.points["up"]["relative_y"],
            center_y,
            self.points["down"]["relative_y"],
        )

        return {
            "coordinate_system": "shoulder_relative_arm_length_normalized_2d",
            "min_x": min_x,
            "max_x": max_x,
            "min_y": min_y,
            "max_y": max_y,
            "center_x": center_x,
            "center_y": center_y,
            "width": max_x - min_x,
            "height": max_y - min_y,
        }

    def get_usable_workspace(self):
        """
        Return a conservative inner workspace.

        Each directional extent is reduced independently toward the neutral
        position. This preserves left/right and up/down asymmetry.
        """
        workspace = self.get_workspace()

        if workspace is None:
            return None

        cx = workspace["center_x"]
        cy = workspace["center_y"]
        ratio = self.usable_ratio

        min_x = cx - (
            cx - workspace["min_x"]
        ) * ratio
        max_x = cx + (
            workspace["max_x"] - cx
        ) * ratio
        min_y = cy - (
            cy - workspace["min_y"]
        ) * ratio
        max_y = cy + (
            workspace["max_y"] - cy
        ) * ratio

        return {
            "coordinate_system": workspace["coordinate_system"],
            "usable_ratio": ratio,
            "min_x": min_x,
            "max_x": max_x,
            "min_y": min_y,
            "max_y": max_y,
            "center_x": cx,
            "center_y": cy,
            "width": max_x - min_x,
            "height": max_y - min_y,
        }

    def get_usable_cardinal_points(self):
        """Return the inner reach envelope as up/right/down/left points."""
        workspace = self.get_usable_workspace()

        if workspace is None:
            return None

        cx = workspace["center_x"]
        cy = workspace["center_y"]

        return {
            "center": (cx, cy),
            "left": (workspace["min_x"], cy),
            "right": (workspace["max_x"], cy),
            "up": (cx, workspace["min_y"]),
            "down": (cx, workspace["max_y"]),
        }

    def map_target_relative(
        self,
        normalized_x,
        normalized_y,
    ):
        """
        Map a logical 0..1 target into the conservative usable workspace.

        The mapping is neutral-centered and asymmetric: each direction uses
        the participant's own calibrated extent. Logical target positions are
        expected to remain inside a diamond-like envelope around the center;
        points outside it are conservatively projected back onto that envelope.
        """
        workspace = self.get_usable_workspace()

        if workspace is None:
            return None

        signed_x = 2.0 * float(normalized_x) - 1.0
        signed_y = 2.0 * float(normalized_y) - 1.0

        l1_radius = abs(signed_x) + abs(signed_y)
        if l1_radius > 1.0:
            signed_x /= l1_radius
            signed_y /= l1_radius

        cx = workspace["center_x"]
        cy = workspace["center_y"]

        if signed_x < 0:
            relative_x = cx + signed_x * (
                cx - workspace["min_x"]
            )
        else:
            relative_x = cx + signed_x * (
                workspace["max_x"] - cx
            )

        if signed_y < 0:
            relative_y = cy + signed_y * (
                cy - workspace["min_y"]
            )
        else:
            relative_y = cy + signed_y * (
                workspace["max_y"] - cy
            )

        return relative_x, relative_y


class BilateralWorkspaceManager:
    """Keep independent shoulder-relative calibrations for both arms."""

    def __init__(
        self,
        capture_duration=0.7,
        min_samples=8,
        min_visibility=0.55,
        usable_ratio=DEFAULT_USABLE_RATIO,
    ):
        self.calibrators = {
            arm: ArmWorkspaceCalibrator(
                capture_duration=capture_duration,
                min_samples=min_samples,
                min_visibility=min_visibility,
                usable_ratio=usable_ratio,
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
