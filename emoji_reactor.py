#!/usr/bin/env python3
"""
Unified Emoji + Monkey Reactor

Brings together the original emoji reactor (plain idle) and the monkey gesture
experience with extra emoji reactions:

- Idle: chill plain image
- Finger to mouth: shh monkey still
- Raised index finger: pointing monkey still
- Thumbs up: thumbs-up PNG
- Tongue out / tongue wiggle: animated monkey GIF
- Big smile (teeth showing): smile.jpg from emoji reactor
- Hands on/above head: air.jpg from emoji reactor

MediaPipe Hands + FaceMesh power all detections so every legacy gesture works
in a single program.
"""

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
from collections import deque
import time
import os

# --- SETUP AND INITIALIZATION ---

# Initialize MediaPipe modules
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose

# --- CONFIGURATION CONSTANTS ---
# Display/window configuration
WINDOW_WIDTH = 720
WINDOW_HEIGHT = 450
EMOJI_WINDOW_SIZE = (WINDOW_WIDTH, WINDOW_HEIGHT)

# Performance settings
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
MONKEY_GIF_FPS = 10

# Monkey gesture thresholds
FINGER_MOUTH_DISTANCE = 0.15
INDEX_EXTENDED_DELTA = 0.10
INDEX_HIGH_DELTA = 0.15
MIDDLE_RELAX_DELTA = 0.05
MOUTH_OPEN_THRESHOLD = 0.02
TONGUE_HISTORY_SIZE = 10
TONGUE_MOVEMENT_THRESHOLD = 0.015
SMILE_ASPECT_THRESHOLD = 0.32
HANDS_ON_HEAD_Y_THRESHOLD = 0.55
HANDS_POSE_MARGIN = 0.02
THUMB_UP_DELTA = 0.08

# Asset paths
PLAIN_IMAGE_PATH = "images/plain.png"
MONKEY_FINGER_MOUTH_PATH = "images/monkey_finger_mouth.jpeg"
MONKEY_FINGER_RAISE_PATH = "images/monkey_finger_raise.jpg"
MONKEY_TONGUE_GIF_PATH = "images/monkey_mouth.gif"
SMILE_IMAGE_PATH = "images/smile.jpg"
AIR_IMAGE_PATH = "images/air.jpg"
THUMBS_IMAGE_PATH = "images/thumbsup.png"

# Helper function to load GIF frames
def load_gif_frames(gif_path):
    gif = Image.open(gif_path)
    frames = []
    try:
        while True:
            frame = gif.convert('RGB')
            frame_cv = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
            frame_resized = cv2.resize(frame_cv, EMOJI_WINDOW_SIZE)
            frames.append(frame_resized)
            gif.seek(gif.tell() + 1)
    except EOFError:
        pass
    return frames

# Helper function to load static images
def load_static_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"{image_path} could not be loaded")
    img_resized = cv2.resize(img, EMOJI_WINDOW_SIZE)
    return img_resized

def try_load_static_image(path, fallback_image, label):
    try:
        if path and os.path.exists(path):
            return load_static_image(path)
        raise FileNotFoundError(f"{path} missing on disk")
    except Exception as exc:
        print(f"⚠️  {label} asset unavailable ({exc}). Using plain fallback.")
        return fallback_image.copy()


def try_load_gif_frames(path, label):
    if not path or not os.path.exists(path):
        print(f"⚠️  {label} GIF not found. Animation disabled.")
        return []
    try:
        frames = load_gif_frames(path)
        if frames:
            return frames
        raise FileNotFoundError("No frames decoded")
    except Exception as exc:
        print(f"⚠️  {label} GIF failed to load ({exc}). Animation disabled.")
        return []


# --- LOAD AND PREPARE IMAGES AND ANIMATIONS ---
try:
    plain_idle_image = load_static_image(PLAIN_IMAGE_PATH)
except Exception as e:
    print("❌ Error: plain idle image is required.")
    print(f"Details: {e}")
    exit()

fallback_image = plain_idle_image.copy()

monkey_finger_mouth_image = try_load_static_image(MONKEY_FINGER_MOUTH_PATH, fallback_image, "Finger-mouth")
monkey_finger_raise_image = try_load_static_image(MONKEY_FINGER_RAISE_PATH, fallback_image, "Finger-raise")
smile_image = try_load_static_image(SMILE_IMAGE_PATH, fallback_image, "Smile")
air_image = try_load_static_image(AIR_IMAGE_PATH, fallback_image, "Hands up")
thumbs_image = try_load_static_image(THUMBS_IMAGE_PATH, fallback_image, "Thumbs up")
monkey_mouth_frames = try_load_gif_frames(MONKEY_TONGUE_GIF_PATH, "Tongue out")

print("✅ Asset load complete.")

# --- MAIN LOGIC ---

# Start webcam capture
print("🎥 Starting webcam capture...")
cap = cv2.VideoCapture(0)

# Set camera resolution for better performance
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, 30)

# Check if webcam is available
if not cap.isOpened():
    print("❌ Error: Could not open webcam. Make sure your camera is connected and not being used by another application.")
    exit()

# Initialize named windows with specific sizes
cv2.namedWindow('Camera Feed', cv2.WINDOW_NORMAL)
cv2.namedWindow('Animation Output', cv2.WINDOW_NORMAL)

# Set window sizes and positions
cv2.resizeWindow('Camera Feed', WINDOW_WIDTH, WINDOW_HEIGHT)
cv2.resizeWindow('Animation Output', WINDOW_WIDTH, WINDOW_HEIGHT)

# Position windows side by side
cv2.moveWindow('Camera Feed', 100, 100)
cv2.moveWindow('Animation Output', WINDOW_WIDTH + 150, 100)

print("🚀 Unified Emoji + Monkey Reactor ready!")
print("📋 Gestures:")
print("   - Finger to mouth → Shh monkey")
print("   - Single raised index finger → Pointing monkey")
print("   - Tongue out / wiggle → Animated monkey GIF")
print("   - Big smile with teeth → Smile image")
print("   - Hands on/above head → Air image")
print("   - Thumbs up → Thumbs PNG")
print("   - Idle → Plain chill image")
print("   - Press 'q' to quit")

# Animation + detection state
current_animation = "EMOJI_IDLE"
previous_monkey_state = "EMOJI_IDLE"
monkey_gif_index = 0
monkey_frame_delay = 1.0 / MONKEY_GIF_FPS
last_monkey_frame_time = time.time()
tongue_x_history = deque(maxlen=TONGUE_HISTORY_SIZE)

def detect_monkey_state(hand_landmarks_list, face_landmarks, pose_landmarks):
    global previous_monkey_state  # noqa: PLW0603 (handled intentionally)

    mouth_info = None
    if face_landmarks:
        upper_lip = face_landmarks.landmark[13]
        lower_lip = face_landmarks.landmark[14]
        mouth_left = face_landmarks.landmark[61]
        mouth_right = face_landmarks.landmark[291]
        mouth_center_x = (mouth_left.x + mouth_right.x) / 2
        mouth_center_y = (upper_lip.y + lower_lip.y) / 2
        mouth_height = ((lower_lip.x - upper_lip.x) ** 2 + (lower_lip.y - upper_lip.y) ** 2) ** 0.5
        mouth_width = ((mouth_right.x - mouth_left.x) ** 2 + (mouth_right.y - mouth_left.y) ** 2) ** 0.5
        mouth_aspect_ratio = mouth_height / mouth_width if mouth_width else 0
        mouth_info = {
            "center_x": mouth_center_x,
            "center_y": mouth_center_y,
            "height": mouth_height,
            "width": mouth_width,
            "aspect_ratio": mouth_aspect_ratio
        }

    # 1. Finger to mouth (highest priority)
    if hand_landmarks_list and mouth_info:
        for hand_landmarks in hand_landmarks_list:
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            distance = ((index_tip.x - mouth_info["center_x"]) ** 2 + (index_tip.y - mouth_info["center_y"]) ** 2) ** 0.5
            if distance < FINGER_MOUTH_DISTANCE:
                if previous_monkey_state != "MONKEY_FINGER_MOUTH":
                    print("✅ MONKEY: Finger to mouth detected!")
                previous_monkey_state = "MONKEY_FINGER_MOUTH"
                return "MONKEY_FINGER_MOUTH"

    # 2. Raised finger
    if hand_landmarks_list:
        for hand_landmarks in hand_landmarks_list:
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
            middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

            index_extended = index_tip.y < index_mcp.y - INDEX_EXTENDED_DELTA
            index_high = index_tip.y < wrist.y - INDEX_HIGH_DELTA
            middle_relaxed = middle_tip.y > index_tip.y + MIDDLE_RELAX_DELTA

            if index_extended and index_high and middle_relaxed:
                if previous_monkey_state != "MONKEY_FINGER_RAISE":
                    print("✅ MONKEY: Raised finger detected!")
                previous_monkey_state = "MONKEY_FINGER_RAISE"
                return "MONKEY_FINGER_RAISE"

    # 3. Thumbs up gesture
    if hand_landmarks_list:
        for hand_landmarks in hand_landmarks_list:
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]

            thumb_up = thumb_tip.y < wrist.y - THUMB_UP_DELTA and thumb_tip.y < thumb_ip.y
            fingers_folded = (
                index_tip.y > wrist.y - 0.02 and
                middle_tip.y > wrist.y - 0.02 and
                ring_tip.y > wrist.y - 0.02
            )

            if thumb_up and fingers_folded:
                if previous_monkey_state != "THUMBS_UP":
                    print("👍 Thumbs up detected!")
                previous_monkey_state = "THUMBS_UP"
                return "THUMBS_UP"

    # 4. Smile detection (teeth showing)
    if mouth_info and mouth_info["aspect_ratio"] > SMILE_ASPECT_THRESHOLD:
        if previous_monkey_state != "EMOJI_SMILE":
            print("😁 Emoji: Big smile detected!")
        previous_monkey_state = "EMOJI_SMILE"
        tongue_x_history.clear()
        return "EMOJI_SMILE"

    # 5. Hands on/above head (pose-based with fallback)
    pose_hands_up = False
    if pose_landmarks:
        landmarks = pose_landmarks.landmark
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
        if (
            left_wrist.y < left_shoulder.y - HANDS_POSE_MARGIN
            or right_wrist.y < right_shoulder.y - HANDS_POSE_MARGIN
        ):
            pose_hands_up = True

    hands_up = False
    if hand_landmarks_list:
        wrist_positions = [hand.landmark[mp_hands.HandLandmark.WRIST].y for hand in hand_landmarks_list]
        hands_up = wrist_positions and all(y < HANDS_ON_HEAD_Y_THRESHOLD for y in wrist_positions)

    if pose_hands_up or hands_up:
        if previous_monkey_state != "AIR_HANDS_UP":
            print("🙌 Hands on head detected!")
        previous_monkey_state = "AIR_HANDS_UP"
        tongue_x_history.clear()
        return "AIR_HANDS_UP"

    # 6. Tongue out (via mouth motion)
    if mouth_info and mouth_info["height"] > MOUTH_OPEN_THRESHOLD:
        tongue_x_history.append(mouth_info["center_x"])
        if len(tongue_x_history) >= TONGUE_HISTORY_SIZE:
            x_range = max(tongue_x_history) - min(tongue_x_history)
            if x_range > TONGUE_MOVEMENT_THRESHOLD:
                if previous_monkey_state != "MONKEY_TONGUE_OUT":
                    print("✅ MONKEY: Tongue out detected!")
                previous_monkey_state = "MONKEY_TONGUE_OUT"
                return "MONKEY_TONGUE_OUT"
    else:
        tongue_x_history.clear()

    if previous_monkey_state != "EMOJI_IDLE":
        print("😌 Returning to plain idle")
    previous_monkey_state = "EMOJI_IDLE"
    return "EMOJI_IDLE"

# Instantiate MediaPipe models
with mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6, max_num_hands=2) as hands, \
     mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True,
                           min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh, \
     mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("⚠️  Ignoring empty camera frame.")
            continue

        frame = cv2.flip(frame, 1)

        small_frame = cv2.resize(frame, (320, 240))
        image_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False

        results_hands = hands.process(image_rgb)
        results_face = face_mesh.process(image_rgb)
        results_pose = pose.process(image_rgb)

        hand_landmarks_list = results_hands.multi_hand_landmarks if results_hands and results_hands.multi_hand_landmarks else []
        face_landmarks = results_face.multi_face_landmarks[0] if results_face and results_face.multi_face_landmarks else None
        pose_landmarks = results_pose.pose_landmarks if results_pose and results_pose.pose_landmarks else None

        current_time = time.time()
        current_animation = detect_monkey_state(hand_landmarks_list, face_landmarks, pose_landmarks)

        # --- DISPLAY LOGIC ---
        if current_animation == "MONKEY_FINGER_MOUTH":
            display_frame = monkey_finger_mouth_image
            state_name = "🤫 Finger to Mouth"
        elif current_animation == "MONKEY_FINGER_RAISE":
            display_frame = monkey_finger_raise_image
            state_name = "☝️ Raised Finger"
        elif current_animation == "THUMBS_UP":
            display_frame = thumbs_image
            state_name = "👍 Thumbs Up"
        elif current_animation == "EMOJI_SMILE":
            display_frame = smile_image
            state_name = "😁 Big Smile"
        elif current_animation == "AIR_HANDS_UP":
            display_frame = air_image
            state_name = "🌬️ Hands Up"
        elif current_animation == "MONKEY_TONGUE_OUT":
            if monkey_mouth_frames:
                if current_time - last_monkey_frame_time >= monkey_frame_delay:
                    monkey_gif_index = (monkey_gif_index + 1) % len(monkey_mouth_frames)
                    last_monkey_frame_time = current_time
                display_frame = monkey_mouth_frames[monkey_gif_index]
                state_name = "👅 Tongue Out"
            else:
                display_frame = plain_idle_image
                state_name = "👅 Tongue Out (no GIF)"
        else:
            display_frame = plain_idle_image
            state_name = "😌 Plain Idle"

        camera_frame_resized = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))

        instruction_text = (
            'Gestures: finger-mouth, finger raise, thumbs up, smile, hands on head, tongue | Press "q"'
        )
        cv2.putText(camera_frame_resized, f'STATE: {state_name}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(camera_frame_resized, instruction_text,
                    (10, WINDOW_HEIGHT - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Camera Feed', camera_frame_resized)
        cv2.imshow('Animation Output', display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# --- CLEANUP ---
print("👋 Shutting down...")
cap.release()
cv2.destroyAllWindows()