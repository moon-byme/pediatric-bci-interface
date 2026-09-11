import cv2
import time
import mediapipe as mp

from pathlib import Path
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


WINDOW_NAME = "Upper Limb Tracking"
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720

MODELS_DIRECTORY = Path(__file__).parent / "models"
MODEL_PATHS = {
    "full": MODELS_DIRECTORY / "pose_landmarker_full.task",
    "lite": MODELS_DIRECTORY / "pose_landmarker_lite.task",
}

PREFERRED_MODEL = "full"
DEFAULT_MIN_VISIBILITY = 0.55


def select_model_path(preferred_model=PREFERRED_MODEL):
    """
    Prefer the Full model when it is available, but keep the project runnable
    with the existing Lite model.
    """
    preferred_path = MODEL_PATHS.get(preferred_model)

    if preferred_path is not None and preferred_path.exists():
        return preferred_model, preferred_path

    for variant in ("full", "lite"):
        path = MODEL_PATHS[variant]
        if path.exists():
            return variant, path

    expected = ", ".join(
        str(path) for path in MODEL_PATHS.values()
    )
    raise FileNotFoundError(
        "No Pose Landmarker model was found. Expected one of: "
        f"{expected}"
    )


class PoseTracker:
    """Detect shoulders, elbows and wrists with MediaPipe Pose Landmarker."""

    def __init__(self, preferred_model=PREFERRED_MODEL):
        self.model_variant, self.model_path = select_model_path(
            preferred_model
        )

        base_options = python.BaseOptions(
            model_asset_path=str(self.model_path)
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
    """Mirror only coordinates used for display; keep anatomical names."""
    mirrored = {}

    for name, point in landmarks.items():
        mirrored[name] = point.copy()
        mirrored[name]["x"] = 1.0 - point["x"]
        mirrored[name]["pixel_x"] = (
            frame_width - 1 - point["pixel_x"]
        )

    return mirrored


def get_arm_quality(
    landmarks,
    arm,
    min_visibility=DEFAULT_MIN_VISIBILITY,
):
    """
    Check the three landmarks required for one upper limb.

    No comparison with the opposite arm is made. Each arm is evaluated from
    its own shoulder, elbow and wrist visibility values.
    """
    names = [
        f"{arm}_shoulder",
        f"{arm}_elbow",
        f"{arm}_wrist",
    ]

    missing = [
        name for name in names
        if name not in landmarks
    ]

    if missing:
        return {
            "reliable": False,
            "missing": missing,
            "min_visibility": 0.0,
            "mean_visibility": 0.0,
            "visibilities": {},
        }

    visibilities = {
        name: float(landmarks[name].get("visibility", 0.0))
        for name in names
    }

    values = list(visibilities.values())
    minimum = min(values)
    mean = sum(values) / len(values)

    return {
        "reliable": minimum >= min_visibility,
        "missing": [],
        "min_visibility": minimum,
        "mean_visibility": mean,
        "visibilities": visibilities,
    }


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


def draw_selected_arm(
    frame,
    landmarks,
    arm,
    min_visibility=DEFAULT_MIN_VISIBILITY,
):
    """
    Draw only the arm currently used by the task.

    Low-visibility points are not drawn as valid landmarks. This avoids making
    an inactive or poorly estimated opposite arm look like a requirement for
    the selected arm.
    """
    shoulder_name = f"{arm}_shoulder"
    elbow_name = f"{arm}_elbow"
    wrist_name = f"{arm}_wrist"

    shoulder = landmarks.get(shoulder_name)
    elbow = landmarks.get(elbow_name)
    wrist = landmarks.get(wrist_name)

    shoulder_ok = (
        shoulder is not None
        and shoulder.get("visibility", 0.0) >= min_visibility
    )
    elbow_ok = (
        elbow is not None
        and elbow.get("visibility", 0.0) >= min_visibility
    )
    wrist_ok = (
        wrist is not None
        and wrist.get("visibility", 0.0) >= min_visibility
    )

    if shoulder_ok and elbow_ok:
        draw_line(frame, shoulder, elbow)

    if elbow_ok and wrist_ok:
        draw_line(frame, elbow, wrist)

    if shoulder_ok:
        draw_point(frame, shoulder, radius=9, color=(0, 255, 0))

    if elbow_ok:
        draw_point(frame, elbow, radius=9, color=(0, 255, 0))

    if wrist_ok:
        wrist_color = (
            (0, 0, 255)
            if arm == "right"
            else (255, 0, 0)
        )

        draw_point(
            frame,
            wrist,
            radius=18,
            color=wrist_color,
        )

        draw_label(
            frame,
            f"{arm.capitalize()} wrist",
            wrist,
        )


def draw_upper_limb_skeleton(frame, landmarks):
    """Draw both arms. Kept mainly for the standalone tracking preview."""
    draw_selected_arm(
        frame,
        landmarks,
        "left",
        min_visibility=0.0,
    )
    draw_selected_arm(
        frame,
        landmarks,
        "right",
        min_visibility=0.0,
    )


def run_tracking_preview():
    """Run upper-limb tracking with mirror-like display."""

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
        cv2.WINDOW_NORMAL
        | cv2.WINDOW_FREERATIO,
    )
    cv2.resizeWindow(
        WINDOW_NAME,
        DISPLAY_WIDTH,
        DISPLAY_HEIGHT,
    )

    print()
    print(f"Pose tracking started with {tracker.model_variant} model.")
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

            landmarks = tracker.process_frame(
                frame,
                timestamp_ms,
            )

            display_frame = cv2.flip(
                frame,
                1,
            )

            display_landmarks = mirror_landmarks_for_display(
                landmarks,
                DISPLAY_WIDTH,
            )

            draw_upper_limb_skeleton(
                display_frame,
                display_landmarks,
            )

            cv2.putText(
                display_frame,
                f"Upper Limb Tracking - model: {tracker.model_variant}",
                (25, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
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
