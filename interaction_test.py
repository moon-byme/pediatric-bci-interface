import math
import time

import cv2

from calibration import (
    BilateralWorkspaceManager,
    EMAFilter,
    ShoulderReferenceFilter,
    get_shoulder_relative_state,
    relative_to_screen,
)
from data_logger import TrialLogger
from pose_tracking import (
    DISPLAY_HEIGHT,
    DISPLAY_WIDTH,
    PoseTracker,
    draw_selected_arm,
    get_arm_quality,
    mirror_landmarks_for_display,
    resize_and_crop,
)


WINDOW_NAME = "Upper Limb Interaction Test"
DWELL_TIME = 1.0
MIN_VISIBILITY = 0.55
SMOOTHING_ALPHA = 0.25
REFERENCE_SMOOTHING_ALPHA = 0.20
USABLE_WORKSPACE_RATIO = 0.85
TARGET_RADIUS_ARM_RATIO = 0.12
MIN_TARGET_RADIUS = 40
MAX_TARGET_RADIUS = 72

# Logical positions inside the participant-specific usable workspace.
# These positions stay inside a conservative diamond-like envelope rather
# than assuming the four corners of a rectangular workspace are reachable.
TARGET_POSITIONS = [
    (0.50, 0.20),
    (0.80, 0.50),
    (0.50, 0.80),
    (0.20, 0.50),
    (0.68, 0.32),
    (0.32, 0.68),
    (0.50, 0.50),
]


class InteractionTest:
    def __init__(self, pose_model):
        self.arm = "right"
        self.pose_model = pose_model
        self.target_index = 0
        self.hover_start = None
        self.success_time = None
        self.total_completed_trials = 0
        self.arm_completed_trials = {
            "right": 0,
            "left": 0,
        }
        self.fullscreen = False

        self.logger = TrialLogger()

        self.workspaces = BilateralWorkspaceManager(
            capture_duration=0.7,
            min_samples=8,
            min_visibility=MIN_VISIBILITY,
            usable_ratio=USABLE_WORKSPACE_RATIO,
        )

        self.relative_wrist_filters = {
            "right": EMAFilter(alpha=SMOOTHING_ALPHA),
            "left": EMAFilter(alpha=SMOOTHING_ALPHA),
        }

        self.reference_filters = {
            "right": ShoulderReferenceFilter(
                alpha=REFERENCE_SMOOTHING_ALPHA
            ),
            "left": ShoulderReferenceFilter(
                alpha=REFERENCE_SMOOTHING_ALPHA
            ),
        }

        self.last_quality = self.empty_quality()

    @staticmethod
    def empty_quality():
        return {
            "reliable": False,
            "missing": [],
            "min_visibility": 0.0,
            "mean_visibility": 0.0,
            "visibilities": {},
        }

    @property
    def calibrator(self):
        return self.workspaces.get(self.arm)

    @property
    def relative_wrist_filter(self):
        return self.relative_wrist_filters[self.arm]

    @property
    def reference_filter(self):
        return self.reference_filters[self.arm]

    @property
    def is_calibrated(self):
        return self.workspaces.is_complete(self.arm)

    def reset_filters(self):
        self.relative_wrist_filter.reset()
        self.reference_filter.reset()

    def set_arm(self, arm):
        if arm not in {"left", "right"}:
            return

        if arm == self.arm:
            return

        self.arm = arm
        self.target_index = 0
        self.hover_start = None
        self.success_time = None
        self.reset_filters()

    def restart_current_calibration(self):
        self.workspaces.reset_arm(self.arm)
        self.reset_filters()
        self.target_index = 0
        self.hover_start = None
        self.success_time = None

    def start_calibration_capture(self):
        self.calibrator.start_capture()

    def next_target(self):
        if not self.is_calibrated:
            return

        self.target_index = (
            self.target_index + 1
        ) % len(TARGET_POSITIONS)

        self.hover_start = None
        self.success_time = None

    def get_current_reference(self, relative_state):
        if relative_state is None:
            return None

        return self.reference_filter.update(
            relative_state["shoulder_x"],
            relative_state["shoulder_y"],
            relative_state["arm_scale"],
        )

    def get_target_relative(self):
        if not self.is_calibrated:
            return None

        logical_x, logical_y = TARGET_POSITIONS[
            self.target_index
        ]

        return self.calibrator.map_target_relative(
            logical_x,
            logical_y,
        )

    def get_target_radius(self, reference):
        if reference is None:
            return MIN_TARGET_RADIUS

        radius = int(
            reference["arm_scale"]
            * TARGET_RADIUS_ARM_RATIO
        )

        return max(
            MIN_TARGET_RADIUS,
            min(MAX_TARGET_RADIUS, radius),
        )

    def update_calibration(
        self,
        relative_state,
        arm_quality,
    ):
        event = self.calibrator.update(
            relative_state,
            arm_reliable=arm_quality["reliable"],
        )

        if event == "complete":
            self.reset_filters()

            self.logger.save_arm_calibration(
                arm=self.arm,
                calibration_points=self.calibrator.points,
                comfortable_workspace=(
                    self.calibrator.get_workspace()
                ),
                usable_workspace=(
                    self.calibrator.get_usable_workspace()
                ),
                usable_ratio=USABLE_WORKSPACE_RATIO,
                pose_model=self.pose_model,
            )

        return self.calibrator.capture_progress()

    def update_interaction(
        self,
        relative_state,
        arm_quality,
    ):
        if self.success_time is not None:
            if (
                time.perf_counter()
                - self.success_time
                >= 0.6
            ):
                self.next_target()

            return {
                "progress": 1.0,
                "interaction_point": None,
                "target_center": None,
                "target_relative": None,
                "reference": None,
                "target_radius": MIN_TARGET_RADIUS,
            }

        if (
            not arm_quality["reliable"]
            or relative_state is None
        ):
            self.hover_start = None
            self.reset_filters()

            return {
                "progress": 0.0,
                "interaction_point": None,
                "target_center": None,
                "target_relative": None,
                "reference": None,
                "target_radius": MIN_TARGET_RADIUS,
            }

        reference = self.get_current_reference(
            relative_state
        )

        relative_x, relative_y = (
            self.relative_wrist_filter.update(
                relative_state["relative_x"],
                relative_state["relative_y"],
            )
        )

        interaction_point = relative_to_screen(
            relative_x,
            relative_y,
            reference,
        )

        target_relative = self.get_target_relative()

        if target_relative is None:
            return {
                "progress": 0.0,
                "interaction_point": interaction_point,
                "target_center": None,
                "target_relative": None,
                "reference": reference,
                "target_radius": self.get_target_radius(reference),
            }

        target_relative_x, target_relative_y = (
            target_relative
        )

        target_center = relative_to_screen(
            target_relative_x,
            target_relative_y,
            reference,
        )

        target_radius = self.get_target_radius(
            reference
        )

        # Compare in shoulder-relative, arm-length-normalized space.
        # The target radius is converted back to the same normalized unit.
        relative_radius = (
            target_radius
            / max(reference["arm_scale"], 1.0)
        )

        distance = math.hypot(
            relative_x - target_relative_x,
            relative_y - target_relative_y,
        )

        if distance <= relative_radius:
            if self.hover_start is None:
                self.hover_start = time.perf_counter()

            elapsed = (
                time.perf_counter()
                - self.hover_start
            )

            progress = min(
                elapsed / DWELL_TIME,
                1.0,
            )

            if elapsed >= DWELL_TIME:
                self.total_completed_trials += 1
                self.arm_completed_trials[self.arm] += 1
                self.success_time = time.perf_counter()

                self.logger.log_success(
                    trial=self.total_completed_trials,
                    arm=self.arm,
                    arm_trial=(
                        self.arm_completed_trials[self.arm]
                    ),
                    target_index=self.target_index + 1,
                    target_screen_x=target_center[0],
                    target_screen_y=target_center[1],
                    target_relative_x=target_relative_x,
                    target_relative_y=target_relative_y,
                    hold_time_seconds=elapsed,
                    arm_scale_px=reference["arm_scale"],
                    min_arm_visibility=(
                        arm_quality["min_visibility"]
                    ),
                    mean_arm_visibility=(
                        arm_quality["mean_visibility"]
                    ),
                    pose_model=self.pose_model,
                )

            return {
                "progress": progress,
                "interaction_point": interaction_point,
                "target_center": target_center,
                "target_relative": target_relative,
                "reference": reference,
                "target_radius": target_radius,
            }

        self.hover_start = None

        return {
            "progress": 0.0,
            "interaction_point": interaction_point,
            "target_center": target_center,
            "target_relative": target_relative,
            "reference": reference,
            "target_radius": target_radius,
        }

    def update(self, display_landmarks):
        arm_quality = get_arm_quality(
            display_landmarks,
            self.arm,
            min_visibility=MIN_VISIBILITY,
        )
        self.last_quality = arm_quality

        relative_state = None

        if arm_quality["reliable"]:
            relative_state = get_shoulder_relative_state(
                display_landmarks,
                self.arm,
            )

        if not self.is_calibrated:
            calibration_progress = self.update_calibration(
                relative_state,
                arm_quality,
            )

            reference = None
            if relative_state is not None:
                reference = {
                    "shoulder_x": relative_state["shoulder_x"],
                    "shoulder_y": relative_state["shoulder_y"],
                    "arm_scale": relative_state["arm_scale"],
                }

            return {
                "mode": "calibration",
                "progress": calibration_progress,
                "interaction_point": None,
                "target_center": None,
                "target_relative": None,
                "target_radius": MIN_TARGET_RADIUS,
                "quality": arm_quality,
                "reference": reference,
            }

        interaction = self.update_interaction(
            relative_state,
            arm_quality,
        )

        interaction.update(
            {
                "mode": "interaction",
                "quality": arm_quality,
            }
        )

        return interaction

    def draw_tracking_status(self, frame, quality, y=145):
        if quality["reliable"]:
            text = (
                "Selected-arm tracking: OK | "
                f"min visibility: {quality['min_visibility']:.2f}"
            )
            color = (120, 255, 120)
        else:
            text = (
                "Selected-arm tracking unstable: keep shoulder, elbow "
                "and wrist visible"
            )
            color = (80, 180, 255)

        cv2.putText(
            frame,
            text,
            (30, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            color,
            2,
            cv2.LINE_AA,
        )

    def draw_relative_calibration_points(
        self,
        frame,
        reference,
    ):
        if reference is None:
            return

        point_colors = {
            "center": (255, 255, 0),
            "left": (255, 180, 0),
            "right": (0, 180, 255),
            "up": (180, 255, 0),
            "down": (255, 0, 180),
        }

        for name, point in self.calibrator.points.items():
            screen_point = relative_to_screen(
                point["relative_x"],
                point["relative_y"],
                reference,
            )

            cv2.circle(
                frame,
                screen_point,
                12,
                point_colors.get(
                    name,
                    (255, 255, 255),
                ),
                3,
            )

            cv2.putText(
                frame,
                name,
                (
                    screen_point[0] + 14,
                    screen_point[1] - 10,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    def draw_workspace_polygon(
        self,
        frame,
        reference,
        usable,
        color,
        label,
    ):
        if reference is None:
            return

        if usable:
            points = self.calibrator.get_usable_cardinal_points()
        else:
            workspace = self.calibrator.get_workspace()
            if workspace is None:
                return

            cx = workspace["center_x"]
            cy = workspace["center_y"]
            points = {
                "up": (cx, workspace["min_y"]),
                "right": (workspace["max_x"], cy),
                "down": (cx, workspace["max_y"]),
                "left": (workspace["min_x"], cy),
            }

        if points is None:
            return

        polygon = []
        for name in ("up", "right", "down", "left"):
            relative_x, relative_y = points[name]
            polygon.append(
                relative_to_screen(
                    relative_x,
                    relative_y,
                    reference,
                )
            )

        for index in range(len(polygon)):
            point_a = polygon[index]
            point_b = polygon[
                (index + 1) % len(polygon)
            ]

            cv2.line(
                frame,
                point_a,
                point_b,
                color,
                2,
            )

        label_x = min(point[0] for point in polygon)
        label_y = min(point[1] for point in polygon)

        cv2.putText(
            frame,
            label,
            (
                max(20, label_x),
                max(25, label_y - 10),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            color,
            1,
            cv2.LINE_AA,
        )

    def draw_calibration(
        self,
        frame,
        display_landmarks,
        state,
    ):
        quality = state["quality"]
        progress = state["progress"]
        reference = state["reference"]

        draw_selected_arm(
            frame,
            display_landmarks,
            self.arm,
            min_visibility=MIN_VISIBILITY,
        )

        cv2.rectangle(
            frame,
            (15, 15),
            (DISPLAY_WIDTH - 15, 165),
            (20, 20, 20),
            -1,
        )

        title = (
            f"Calibration {self.calibrator.stage_number}/"
            f"{self.calibrator.stage_count} - {self.arm} arm"
        )

        cv2.putText(
            frame,
            title,
            (30, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.82,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            self.calibrator.prompt,
            (30, 78),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            self.calibrator.last_status,
            (30, 108),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.53,
            (180, 230, 255),
            2,
            cv2.LINE_AA,
        )

        self.draw_tracking_status(
            frame,
            quality,
            y=140,
        )

        self.draw_relative_calibration_points(
            frame,
            reference,
        )

        if self.calibrator.capturing:
            bar_x = 30
            bar_y = DISPLAY_HEIGHT - 55
            bar_width = 360
            bar_height = 22

            cv2.rectangle(
                frame,
                (bar_x, bar_y),
                (
                    bar_x + bar_width,
                    bar_y + bar_height,
                ),
                (255, 255, 255),
                2,
            )

            filled = int(bar_width * progress)

            cv2.rectangle(
                frame,
                (bar_x, bar_y),
                (
                    bar_x + filled,
                    bar_y + bar_height,
                ),
                (0, 220, 0),
                -1,
            )

        right_status = (
            "calibrated"
            if self.workspaces.is_complete("right")
            else "not calibrated"
        )
        left_status = (
            "calibrated"
            if self.workspaces.is_complete("left")
            else "not calibrated"
        )

        cv2.putText(
            frame,
            f"Right: {right_status} | Left: {left_status}",
            (30, DISPLAY_HEIGHT - 82),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        controls = (
            "C: capture | R/L: select arm | X: reset selected arm | "
            "F: fullscreen | Q: quit"
        )

        cv2.putText(
            frame,
            controls,
            (30, DISPLAY_HEIGHT - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    def draw_interaction(
        self,
        frame,
        display_landmarks,
        state,
    ):
        quality = state["quality"]
        reference = state["reference"]
        interaction_point = state["interaction_point"]
        target_center = state["target_center"]
        target_radius = state["target_radius"]
        progress = state["progress"]

        draw_selected_arm(
            frame,
            display_landmarks,
            self.arm,
            min_visibility=MIN_VISIBILITY,
        )

        # Raw comfortable reach envelope.
        self.draw_workspace_polygon(
            frame,
            reference,
            usable=False,
            color=(255, 255, 0),
            label=(
                f"{self.arm.capitalize()} comfortable workspace"
            ),
        )

        # Conservative inner envelope used to place game targets.
        self.draw_workspace_polygon(
            frame,
            reference,
            usable=True,
            color=(0, 255, 120),
            label="Usable workspace (85%)",
        )

        if target_center is not None:
            cv2.circle(
                frame,
                target_center,
                target_radius,
                (60, 190, 255),
                4,
            )

            if progress > 0:
                start_angle = -90
                end_angle = int(
                    start_angle + 360 * progress
                )

                cv2.ellipse(
                    frame,
                    target_center,
                    (
                        target_radius + 12,
                        target_radius + 12,
                    ),
                    0,
                    start_angle,
                    end_angle,
                    (0, 255, 0),
                    8,
                )

        if interaction_point is not None:
            cv2.circle(
                frame,
                interaction_point,
                22,
                (255, 255, 255),
                3,
            )

        if reference is not None:
            shoulder_point = (
                int(round(reference["shoulder_x"])),
                int(round(reference["shoulder_y"])),
            )
            cv2.circle(
                frame,
                shoulder_point,
                12,
                (255, 0, 255),
                2,
            )
            cv2.putText(
                frame,
                "shoulder origin",
                (
                    shoulder_point[0] + 15,
                    shoulder_point[1] - 12,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (255, 0, 255),
                1,
                cv2.LINE_AA,
            )

        cv2.rectangle(
            frame,
            (15, 15),
            (DISPLAY_WIDTH - 15, 132),
            (20, 20, 20),
            -1,
        )

        title = (
            f"{self.arm.capitalize()} arm: shoulder-relative target control | "
            f"hold {DWELL_TIME:.1f}s"
        )

        cv2.putText(
            frame,
            title,
            (30, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        controls = (
            "R/L: switch arm | SPACE: next target | X: recalibrate | "
            "F: fullscreen | Q: quit"
        )

        cv2.putText(
            frame,
            controls,
            (30, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        self.draw_tracking_status(
            frame,
            quality,
            y=108,
        )

        counter_text = (
            f"Total: {self.total_completed_trials} | "
            f"Right: {self.arm_completed_trials['right']} | "
            f"Left: {self.arm_completed_trials['left']} | "
            f"Model: {self.pose_model} | Useful: {int(USABLE_WORKSPACE_RATIO * 100)}%"
        )

        cv2.putText(
            frame,
            counter_text,
            (25, DISPLAY_HEIGHT - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.54,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        if self.success_time is not None:
            success_text = "Target completed!"

            text_size, _ = cv2.getTextSize(
                success_text,
                cv2.FONT_HERSHEY_SIMPLEX,
                1.1,
                3,
            )

            text_x = (
                DISPLAY_WIDTH - text_size[0]
            ) // 2

            cv2.putText(
                frame,
                success_text,
                (text_x, 175),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.1,
                (0, 255, 0),
                3,
                cv2.LINE_AA,
            )

    def draw(
        self,
        frame,
        display_landmarks,
        state,
    ):
        if state["mode"] == "calibration":
            self.draw_calibration(
                frame,
                display_landmarks,
                state,
            )
            return

        self.draw_interaction(
            frame,
            display_landmarks,
            state,
        )

    def toggle_fullscreen(self):
        self.fullscreen = not self.fullscreen

        mode = (
            cv2.WINDOW_FULLSCREEN
            if self.fullscreen
            else cv2.WINDOW_NORMAL
        )

        cv2.setWindowProperty(
            WINDOW_NAME,
            cv2.WND_PROP_FULLSCREEN,
            mode,
        )

        if not self.fullscreen:
            cv2.resizeWindow(
                WINDOW_NAME,
                DISPLAY_WIDTH,
                DISPLAY_HEIGHT,
            )


def run_interaction_test():
    camera = cv2.VideoCapture(
        0,
        cv2.CAP_DSHOW,
    )

    camera.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        DISPLAY_WIDTH,
    )

    camera.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        DISPLAY_HEIGHT,
    )

    if not camera.isOpened():
        raise RuntimeError(
            "Could not open webcam."
        )

    tracker = PoseTracker()
    test = InteractionTest(
        pose_model=tracker.model_variant
    )
    start_time = time.perf_counter()

    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL
        | cv2.WINDOW_FREERATIO,
    )

    cv2.resizeWindow(
        WINDOW_NAME,
        DISPLAY_WIDTH,
        DISPLAY_HEIGHT,
    )

    print()
    print("Upper-limb interaction test started.")
    print(f"Pose model: {tracker.model_variant}")
    print("Right and left arms are calibrated independently.")
    print("Interaction coordinates are relative to the selected shoulder.")
    print(
        f"Targets use {int(USABLE_WORKSPACE_RATIO * 100)}% "
        "of each directional calibrated extent."
    )
    print(f"Session data: {test.logger.file_path}")
    print()

    try:
        while True:
            success, frame = camera.read()

            if not success:
                print("Could not read webcam frame.")
                break

            frame = resize_and_crop(
                frame,
                DISPLAY_WIDTH,
                DISPLAY_HEIGHT,
            )

            timestamp_ms = int(
                (
                    time.perf_counter()
                    - start_time
                )
                * 1000
            )

            # Anatomical labels are estimated on the original image.
            landmarks = tracker.process_frame(
                frame,
                timestamp_ms,
            )

            # The participant sees a mirror-like image.
            display_frame = cv2.flip(
                frame,
                1,
            )

            display_landmarks = mirror_landmarks_for_display(
                landmarks,
                DISPLAY_WIDTH,
            )

            state = test.update(
                display_landmarks
            )

            test.draw(
                display_frame,
                display_landmarks,
                state,
            )

            cv2.imshow(
                WINDOW_NAME,
                display_frame,
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("r"):
                test.set_arm("right")

            if key == ord("l"):
                test.set_arm("left")

            if key == ord("c"):
                test.start_calibration_capture()

            if key == ord("x"):
                test.restart_current_calibration()

            if key == ord("f"):
                test.toggle_fullscreen()

            if key == 32:
                test.next_target()

    finally:
        camera.release()
        tracker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_interaction_test()
