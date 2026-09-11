import math
import time

import cv2

from calibration import (
    BilateralWorkspaceManager,
    EMAFilter,
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
TARGET_RADIUS = 55
MIN_VISIBILITY = 0.55
SMOOTHING_ALPHA = 0.25

# Logical target positions inside EACH ARM'S independently calibrated workspace.
TARGET_POSITIONS = [
    (0.20, 0.25),
    (0.80, 0.30),
    (0.75, 0.75),
    (0.30, 0.72),
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
        )

        self.wrist_filters = {
            "right": EMAFilter(alpha=SMOOTHING_ALPHA),
            "left": EMAFilter(alpha=SMOOTHING_ALPHA),
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
    def wrist_filter(self):
        return self.wrist_filters[self.arm]

    @property
    def wrist_name(self):
        return f"{self.arm}_wrist"

    @property
    def is_calibrated(self):
        return self.workspaces.is_complete(self.arm)

    def get_target_center(self):
        if not self.is_calibrated:
            return None

        normalized_x, normalized_y = TARGET_POSITIONS[
            self.target_index
        ]

        return self.calibrator.map_target(
            normalized_x,
            normalized_y,
            safe_margin=0.08,
        )

    def set_arm(self, arm):
        if arm not in {"left", "right"}:
            return

        if arm == self.arm:
            return

        self.arm = arm
        self.target_index = 0
        self.hover_start = None
        self.success_time = None
        self.wrist_filter.reset()

    def restart_current_calibration(self):
        self.workspaces.reset_arm(self.arm)
        self.wrist_filter.reset()
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

    def update_calibration(
        self,
        display_landmarks,
        arm_quality,
    ):
        wrist = display_landmarks.get(
            self.wrist_name
        )

        event = self.calibrator.update(
            wrist,
            arm_reliable=arm_quality["reliable"],
        )

        if event == "complete":
            self.wrist_filter.reset()

            self.logger.save_arm_calibration(
                arm=self.arm,
                calibration_points=self.calibrator.points,
                workspace=self.calibrator.get_workspace(),
                pose_model=self.pose_model,
            )

        return self.calibrator.capture_progress()

    def update_interaction(
        self,
        display_landmarks,
        arm_quality,
    ):
        if self.success_time is not None:
            if (
                time.perf_counter()
                - self.success_time
                >= 0.6
            ):
                self.next_target()

            return 1.0, None

        if not arm_quality["reliable"]:
            self.hover_start = None
            self.wrist_filter.reset()
            return 0.0, None

        wrist = display_landmarks.get(
            self.wrist_name
        )

        if wrist is None:
            self.hover_start = None
            self.wrist_filter.reset()
            return 0.0, None

        interaction_x, interaction_y = self.wrist_filter.update(
            wrist["pixel_x"],
            wrist["pixel_y"],
        )

        target_center = self.get_target_center()

        if target_center is None:
            return 0.0, (
                interaction_x,
                interaction_y,
            )

        target_x, target_y = target_center

        distance = math.hypot(
            interaction_x - target_x,
            interaction_y - target_y,
        )

        if distance <= TARGET_RADIUS:
            if self.hover_start is None:
                self.hover_start = time.perf_counter()

            elapsed = time.perf_counter() - self.hover_start
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
                    arm_trial=self.arm_completed_trials[self.arm],
                    target_index=self.target_index + 1,
                    target_x=target_x,
                    target_y=target_y,
                    hold_time_seconds=elapsed,
                    min_arm_visibility=arm_quality["min_visibility"],
                    mean_arm_visibility=arm_quality["mean_visibility"],
                    pose_model=self.pose_model,
                )

            return progress, (
                interaction_x,
                interaction_y,
            )

        self.hover_start = None

        return 0.0, (
            interaction_x,
            interaction_y,
        )

    def update(self, display_landmarks):
        arm_quality = get_arm_quality(
            display_landmarks,
            self.arm,
            min_visibility=MIN_VISIBILITY,
        )
        self.last_quality = arm_quality

        if not self.is_calibrated:
            calibration_progress = self.update_calibration(
                display_landmarks,
                arm_quality,
            )

            return {
                "mode": "calibration",
                "progress": calibration_progress,
                "interaction_point": None,
                "quality": arm_quality,
            }

        dwell_progress, interaction_point = self.update_interaction(
            display_landmarks,
            arm_quality,
        )

        return {
            "mode": "interaction",
            "progress": dwell_progress,
            "interaction_point": interaction_point,
            "quality": arm_quality,
        }

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

    def draw_calibration(
        self,
        frame,
        display_landmarks,
        progress,
        quality,
    ):
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

        point_colors = {
            "center": (255, 255, 0),
            "left": (255, 180, 0),
            "right": (0, 180, 255),
            "up": (180, 255, 0),
            "down": (255, 0, 180),
        }

        for name, point in self.calibrator.points.items():
            cv2.circle(
                frame,
                (point["x"], point["y"]),
                12,
                point_colors.get(name, (255, 255, 255)),
                3,
            )

            cv2.putText(
                frame,
                name,
                (
                    point["x"] + 14,
                    point["y"] - 10,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
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
        progress,
        interaction_point,
        quality,
    ):
        draw_selected_arm(
            frame,
            display_landmarks,
            self.arm,
            min_visibility=MIN_VISIBILITY,
        )

        workspace = self.calibrator.get_workspace()

        if workspace is not None:
            cv2.rectangle(
                frame,
                (
                    workspace["min_x"],
                    workspace["min_y"],
                ),
                (
                    workspace["max_x"],
                    workspace["max_y"],
                ),
                (255, 255, 0),
                2,
            )

            cv2.putText(
                frame,
                f"{self.arm.capitalize()} comfortable workspace",
                (
                    workspace["min_x"],
                    max(25, workspace["min_y"] - 10),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 0),
                1,
                cv2.LINE_AA,
            )

        target_center = self.get_target_center()

        if target_center is not None:
            target_x, target_y = target_center

            cv2.circle(
                frame,
                (target_x, target_y),
                TARGET_RADIUS,
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
                    (target_x, target_y),
                    (
                        TARGET_RADIUS + 12,
                        TARGET_RADIUS + 12,
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
                25,
                (255, 255, 255),
                3,
            )

        cv2.rectangle(
            frame,
            (15, 15),
            (DISPLAY_WIDTH - 15, 132),
            (20, 20, 20),
            -1,
        )

        title = (
            f"{self.arm.capitalize()} arm: move the wrist to the target "
            f"and hold for {DWELL_TIME:.1f}s"
        )

        cv2.putText(
            frame,
            title,
            (30, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        controls = (
            "R/L: switch arm | SPACE: next target | X: recalibrate selected arm | "
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
            f"Model: {self.pose_model}"
        )

        cv2.putText(
            frame,
            counter_text,
            (25, DISPLAY_HEIGHT - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
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
                state["progress"],
                state["quality"],
            )
            return

        self.draw_interaction(
            frame,
            display_landmarks,
            state["progress"],
            state["interaction_point"],
            state["quality"],
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
    print("Only the selected arm is used for interaction quality checks.")
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
