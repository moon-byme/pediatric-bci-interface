import cv2
import time
import mediapipe as mp

from pathlib import Path
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


WINDOW_NAME = "Upper Limb Tracking"
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720

MODEL_PATH = (
    Path(__file__).parent
    / "models"
    / "pose_landmarker_lite.task"
)


class PoseTracker:
    """Detects shoulders, elbows and wrists with MediaPipe Pose Landmarker."""

    def __init__(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found: {MODEL_PATH}"
            )

        base_options = python.BaseOptions(
            model_asset_path=str(MODEL_PATH)
        )

        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.landmarker = vision.PoseLandmarker.create_from_options(
            options
        )

        self.landmark_ids = {
            "left_shoulder": 11,
            "right_shoulder": 12,
            "left_elbow": 13,
            "right_elbow": 14,
            "left_wrist": 15,
            "right_wrist": 16,
        }

    def process_frame(self, frame, timestamp_ms):
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame,
        )

        result = self.landmarker.detect_for_video(
            mp_image,
            timestamp_ms,
        )

        return self.extract_upper_limb_landmarks(
            result,
            frame.shape,
        )

    def extract_upper_limb_landmarks(
        self,
        result,
        frame_shape,
    ):
        landmarks = {}

        if not result.pose_landmarks:
            return landmarks

        pose = result.pose_landmarks[0]
        height, width = frame_shape[:2]

        for name, index in self.landmark_ids.items():
            landmark = pose[index]

            landmarks[name] = {
                "x": landmark.x,
                "y": landmark.y,
                "z": landmark.z,
                "pixel_x": int(landmark.x * width),
                "pixel_y": int(landmark.y * height),
                "visibility": landmark.visibility,
            }

        return landmarks

    def close(self):
        self.landmarker.close()


def resize_and_crop(frame, target_width, target_height):
    """Resize while preserving aspect ratio, then crop from the center."""

    original_height, original_width = frame.shape[:2]

    scale_width = target_width / original_width
    scale_height = target_height / original_height
    scale = max(scale_width, scale_height)

    new_width = int(original_width * scale)
    new_height = int(original_height * scale)

    resized = cv2.resize(
        frame,
        (new_width, new_height),
        interpolation=cv2.INTER_LINEAR,
    )

    x_start = (new_width - target_width) // 2
    y_start = (new_height - target_height) // 2

    return resized[
        y_start:y_start + target_height,
        x_start:x_start + target_width,
    ]


def mirror_landmarks_for_display(landmarks, frame_width):
    """Mirror only landmark coordinates used for drawing."""
    mirrored = {}

    for name, point in landmarks.items():
        mirrored[name] = point.copy()
        mirrored[name]["x"] = 1.0 - point["x"]
        mirrored[name]["pixel_x"] = (
            frame_width - 1 - point["pixel_x"]
        )

    return mirrored


def draw_point(
    frame,
    point,
    radius=10,
    color=(0, 255, 0),
):
    cv2.circle(
        frame,
        (
            point["pixel_x"],
            point["pixel_y"],
        ),
        radius,
        color,
        -1,
    )


def draw_line(
    frame,
    point_a,
    point_b,
    color=(255, 255, 255),
    thickness=4,
):
    cv2.line(
        frame,
        (
            point_a["pixel_x"],
            point_a["pixel_y"],
        ),
        (
            point_b["pixel_x"],
            point_b["pixel_y"],
        ),
        color,
        thickness,
    )


def draw_label(
    frame,
    text,
    point,
    offset_x=10,
    offset_y=-10,
):
    cv2.putText(
        frame,
        text,
        (
            point["pixel_x"] + offset_x,
            point["pixel_y"] + offset_y,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def draw_upper_limb_skeleton(frame, landmarks):
    """Draw both upper limbs."""

    connections = [
        ("left_shoulder", "left_elbow"),
        ("left_elbow", "left_wrist"),
        ("right_shoulder", "right_elbow"),
        ("right_elbow", "right_wrist"),
    ]

    for point_a_name, point_b_name in connections:
        if (
            point_a_name in landmarks
            and point_b_name in landmarks
        ):
            draw_line(
                frame,
                landmarks[point_a_name],
                landmarks[point_b_name],
            )

    for point_name in [
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
    ]:
        if point_name in landmarks:
            draw_point(
                frame,
                landmarks[point_name],
                radius=9,
                color=(0, 255, 0),
            )

    if "left_wrist" in landmarks:
        draw_point(
            frame,
            landmarks["left_wrist"],
            radius=18,
            color=(255, 0, 0),
        )
        draw_label(
            frame,
            "Left wrist",
            landmarks["left_wrist"],
        )

    if "right_wrist" in landmarks:
        draw_point(
            frame,
            landmarks["right_wrist"],
            radius=18,
            color=(0, 0, 255),
        )
        draw_label(
            frame,
            "Right wrist",
            landmarks["right_wrist"],
        )


def run_tracking_preview():
    """Run upper-limb tracking with mirrored view and readable labels."""

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
        raise RuntimeError("Could not open webcam.")

    tracker = PoseTracker()
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
    print("Pose tracking started.")
    print("Press Q to quit.")
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
                (time.perf_counter() - start_time)
                * 1000
            )

            # Analyze the original frame so left/right keep anatomical meaning.
            landmarks = tracker.process_frame(
                frame,
                timestamp_ms,
            )

            # Mirror only the image shown to the user.
            display_frame = cv2.flip(
                frame,
                1,
            )

            # Mirror only landmark X coordinates for drawing.
            display_landmarks = mirror_landmarks_for_display(
                landmarks,
                DISPLAY_WIDTH,
            )

            # Draw after mirroring so labels stay readable.
            draw_upper_limb_skeleton(
                display_frame,
                display_landmarks,
            )

            cv2.putText(
                display_frame,
                "Upper Limb Tracking",
                (25, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                display_frame,
                "Press Q to quit",
                (25, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(
                WINDOW_NAME,
                display_frame,
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

    finally:
        camera.release()
        tracker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_tracking_preview()
