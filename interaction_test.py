import math
import time

import cv2

from data_logger import TrialLogger
from pose_tracking import (
    DISPLAY_HEIGHT,
    DISPLAY_WIDTH,
    PoseTracker,
    draw_upper_limb_skeleton,
    mirror_landmarks_for_display,
    resize_and_crop,
)


WINDOW_NAME = "Upper Limb Interaction Test"
DWELL_TIME = 1.0
TARGET_RADIUS = 55
MIN_VISIBILITY = 0.5

# Engineering-test positions only.
# Later these should be individualized from each participant's
# comfortable range of motion.
TARGET_POSITIONS = [
    (0.72, 0.35),
    (0.78, 0.52),
    (0.62, 0.62),
    (0.55, 0.40),
]


class InteractionTest:
    def __init__(self):
        self.arm = "right"
        self.target_index = 0
        self.hover_start = None
        self.success_time = None
        self.completed_trials = 0
        self.fullscreen = False
        self.logger = TrialLogger()

    @property
    def wrist_name(self):
        return f"{self.arm}_wrist"

    def get_target_center(self):
        normalized_x, normalized_y = TARGET_POSITIONS[
            self.target_index
        ]

        return (
            int(normalized_x * DISPLAY_WIDTH),
            int(normalized_y * DISPLAY_HEIGHT),
        )

    def set_arm(self, arm):
        if arm not in {"left", "right"}:
            return

        self.arm = arm
        self.hover_start = None
        self.success_time = None

    def reset_current_target(self):
        self.hover_start = None
        self.success_time = None

    def next_target(self):
        self.target_index = (
            self.target_index + 1
        ) % len(TARGET_POSITIONS)

        self.hover_start = None
        self.success_time = None

    def update(self, display_landmarks):
        """
        Update interaction using DISPLAY coordinates.

        The anatomical landmark names remain unchanged:
        right_wrist is still the participant's real right wrist,
        but its X coordinate has already been mirrored for the
        on-screen interaction.
        """

        if self.success_time is not None:
            if (
                time.perf_counter()
                - self.success_time
                >= 0.6
            ):
                self.next_target()

            return 1.0

        wrist = display_landmarks.get(
            self.wrist_name
        )

        if (
            wrist is None
            or wrist["visibility"] < MIN_VISIBILITY
        ):
            self.hover_start = None
            return 0.0

        target_x, target_y = (
            self.get_target_center()
        )

        delta_x = (
            wrist["pixel_x"]
            - target_x
        )

        delta_y = (
            wrist["pixel_y"]
            - target_y
        )

        distance = math.hypot(
            delta_x,
            delta_y,
        )

        if distance <= TARGET_RADIUS:
            if self.hover_start is None:
                self.hover_start = (
                    time.perf_counter()
                )

            elapsed = (
                time.perf_counter()
                - self.hover_start
            )

            progress = min(
                elapsed / DWELL_TIME,
                1.0,
            )

            if elapsed >= DWELL_TIME:
                self.completed_trials += 1
                self.success_time = (
                    time.perf_counter()
                )

                self.logger.log_success(
                    trial=self.completed_trials,
                    arm=self.arm,
                    target_x=target_x,
                    target_y=target_y,
                    hold_time_seconds=elapsed,
                )

            return progress

        self.hover_start = None
        return 0.0

    def draw(
        self,
        display_frame,
        display_landmarks,
        progress,
    ):
        # Draw skeleton after the camera image is mirrored,
        # so labels remain readable.
        draw_upper_limb_skeleton(
            display_frame,
            display_landmarks,
        )

        target_x, target_y = (
            self.get_target_center()
        )

        cv2.circle(
            display_frame,
            (target_x, target_y),
            TARGET_RADIUS,
            (60, 190, 255),
            4,
        )

        if progress > 0:
            start_angle = -90
            end_angle = int(
                start_angle
                + 360 * progress
            )

            cv2.ellipse(
                display_frame,
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

        title = (
            f"Move your {self.arm} wrist "
            f"to the target and hold for "
            f"{DWELL_TIME:.1f}s"
        )

        cv2.putText(
            display_frame,
            title,
            (25, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.78,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        controls = (
            "R: right arm | L: left arm | "
            "SPACE: next target | F: fullscreen | Q: quit"
        )

        cv2.putText(
            display_frame,
            controls,
            (25, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        counter_text = (
            f"Completed targets: "
            f"{self.completed_trials}"
        )

        cv2.putText(
            display_frame,
            counter_text,
            (25, DISPLAY_HEIGHT - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
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
                DISPLAY_WIDTH
                - text_size[0]
            ) // 2

            cv2.putText(
                display_frame,
                success_text,
                (text_x, 125),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.1,
                (0, 255, 0),
                3,
                cv2.LINE_AA,
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
    test = InteractionTest()
    start_time = time.perf_counter()

    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO,
    )

    cv2.resizeWindow(
        WINDOW_NAME,
        DISPLAY_WIDTH,
        DISPLAY_HEIGHT,
    )

    print()
    print(
        "Upper-limb interaction test started."
    )
    print(
        "The camera view is mirrored only for display."
    )
    print(
        "Left/right labels preserve anatomical meaning."
    )
    print(
        f"Session data: {test.logger.file_path}"
    )
    print()

    try:
        while True:
            success, frame = camera.read()

            if not success:
                print(
                    "Could not read webcam frame."
                )
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

            # 1. Analyze ORIGINAL frame.
            landmarks = tracker.process_frame(
                frame,
                timestamp_ms,
            )

            # 2. Mirror only the camera image shown to the user.
            display_frame = cv2.flip(
                frame,
                1,
            )

            # 3. Mirror X coordinates only for drawing/interaction.
            display_landmarks = (
                mirror_landmarks_for_display(
                    landmarks,
                    DISPLAY_WIDTH,
                )
            )

            # 4. Interaction uses the same coordinate system
            #    that the participant sees on screen.
            progress = test.update(
                display_landmarks
            )

            # 5. Draw everything AFTER the mirror,
            #    keeping text readable.
            test.draw(
                display_frame,
                display_landmarks,
                progress,
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
