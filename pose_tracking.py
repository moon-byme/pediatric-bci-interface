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
            min_tracking_confidence=0.5
        )

        self.landmarker = (
            vision.PoseLandmarker.create_from_options(
                options
            )
        )

        self.landmark_ids = {
            "left_shoulder": 11,
            "right_shoulder": 12,
            "left_elbow": 13,
            "right_elbow": 14,
            "left_wrist": 15,
            "right_wrist": 16
        }

    def process_frame(
        self,
        frame,
        timestamp_ms
    ):

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        result = self.landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        return self.extract_upper_limb_landmarks(
            result,
            frame.shape
        )

    def extract_upper_limb_landmarks(
        self,
        result,
        frame_shape
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

                "pixel_x": int(
                    landmark.x * width
                ),

                "pixel_y": int(
                    landmark.y * height
                ),

                "visibility": landmark.visibility
            }

        return landmarks

    def close(self):

        self.landmarker.close()

def resize_and_crop(
    frame,
    target_width,
    target_height
):

    original_height, original_width = (
        frame.shape[:2]
    )

    scale_width = (
        target_width / original_width
    )

    scale_height = (
        target_height / original_height
    )

    scale = max(
        scale_width,
        scale_height
    )

    new_width = int(
        original_width * scale
    )

    new_height = int(
        original_height * scale
    )

    resized = cv2.resize(
        frame,
        (new_width, new_height),
        interpolation=cv2.INTER_LINEAR
    )

    x_start = (
        new_width - target_width
    ) 

    y_start = (
        new_height - target_height
    ) 

    cropped = resized[
        y_start:y_start + target_height,
        x_start:x_start + target_width
    ]

    return cropped

def draw_point(
    frame,
    point,
    radius=10,
    color=(0, 255, 0)
):

    cv2.circle(
        frame,
        (
            point["pixel_x"],
            point["pixel_y"]
        ),
        radius,
        color,
        -1
    )


def draw_line(
    frame,
    point_a,
    point_b,
    color=(255, 255, 255),
    thickness=4
):

    cv2.line(
        frame,
        (
            point_a["pixel_x"],
            point_a["pixel_y"]
        ),
        (
            point_b["pixel_x"],
            point_b["pixel_y"]
        ),
        color,
        thickness
    )


def draw_label(
    frame,
    text,
    point,
    offset_x=10,
    offset_y=-10
):

    cv2.putText(
        frame,
        text,
        (
            point["pixel_x"] + offset_x,
            point["pixel_y"] + offset_y
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

if __name__ == "__main__":

    camera = cv2.VideoCapture(
        0,
        cv2.CAP_DSHOW
    )

    camera.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        1280
    )

    camera.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        720
    )

    if not camera.isOpened():

        raise RuntimeError(
            "Could not open webcam."
        )

    actual_width = int(
        camera.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    actual_height = int(
        camera.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    print(
        "Camera resolution:",
        actual_width,
        "x",
        actual_height
    )

    tracker = PoseTracker()

    start_time = time.perf_counter()

    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL
    )

    cv2.resizeWindow(
        WINDOW_NAME,
        DISPLAY_WIDTH,
        DISPLAY_HEIGHT
    )

    cv2.setWindowProperty(
        WINDOW_NAME,
        cv2.WND_PROP_FULLSCREEN,
        cv2.WINDOW_NORMAL
    )

    print()
    print("Pose tracking started.")
    print("Press Q to quit.")
    print()

    try:

        while True:

            success, frame = camera.read()

            if not success:

                print(
                    "Could not read webcam frame."
                )

                break

            frame = cv2.flip(
                frame,
                1
            )

            frame = resize_and_crop(
                frame,
                DISPLAY_WIDTH,
                DISPLAY_HEIGHT
            )

            timestamp_ms = int(
                (
                    time.perf_counter()
                    - start_time
                )
                * 1000
            )

            landmarks = (
                tracker.process_frame(
                    frame,
                    timestamp_ms
                )
            )

            if (
                "left_shoulder" in landmarks
                and
                "left_elbow" in landmarks
            ):

                draw_line(
                    frame,
                    landmarks[
                        "left_shoulder"
                    ],
                    landmarks[
                        "left_elbow"
                    ]
                )

            if (
                "left_elbow" in landmarks
                and
                "left_wrist" in landmarks
            ):

                draw_line(
                    frame,
                    landmarks[
                        "left_elbow"
                    ],
                    landmarks[
                        "left_wrist"
                    ]
                )

            if (
                "right_shoulder" in landmarks
                and
                "right_elbow" in landmarks
            ):

                draw_line(
                    frame,
                    landmarks[
                        "right_shoulder"
                    ],
                    landmarks[
                        "right_elbow"
                    ]
                )

            if (
                "right_elbow" in landmarks
                and
                "right_wrist" in landmarks
            ):

                draw_line(
                    frame,
                    landmarks[
                        "right_elbow"
                    ],
                    landmarks[
                        "right_wrist"
                    ]
                )

            for point_name in [
                "left_shoulder",
                "right_shoulder",
                "left_elbow",
                "right_elbow"
            ]:

                if point_name in landmarks:

                    draw_point(
                        frame,
                        landmarks[
                            point_name
                        ],
                        radius=9,
                        color=(0, 255, 0)
                    )

            if "left_wrist" in landmarks:

                draw_point(
                    frame,
                    landmarks[
                        "left_wrist"
                    ],
                    radius=18,
                    color=(255, 0, 0)
                )

                draw_label(
                    frame,
                    "Left wrist",
                    landmarks[
                        "left_wrist"
                    ]
                )

            if "right_wrist" in landmarks:

                wrist = landmarks[
                    "right_wrist"
                ]

                draw_point(
                    frame,
                    wrist,
                    radius=18,
                    color=(0, 0, 255)
                )

                draw_label(
                    frame,
                    "Right wrist",
                    wrist
                )

            cv2.putText(
                frame,
                "Upper Limb Tracking",
                (25, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            cv2.putText(
                frame,
                "Press Q to quit",
                (25, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            cv2.imshow(
                WINDOW_NAME,
                frame
            )

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            if key == ord("q"):
                break

    finally:

        camera.release()

        tracker.close()

        cv2.destroyAllWindows()