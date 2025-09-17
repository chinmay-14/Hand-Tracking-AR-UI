
import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False,
                        max_num_hands=1,
                        min_detection_confidence=0.7,
                        min_tracking_confidence=0.7)

# Colors for UI overlays (reference style)
CYAN = (255, 255, 0)
ORANGE = (0, 180, 255)
WHITE = (255, 255, 255)
RED = (0, 0, 255)
CORE = (0, 255, 180)

def draw_glow_circle(img, center, radius, color, thickness=2, glow=15):
    # Draw outer glow
    for g in range(glow, 0, -3):
        alpha = 0.08 + 0.12 * (g / glow)
        overlay = img.copy()
        cv2.circle(overlay, center, radius+g, color, thickness)
        cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)
    # Draw main circle
    cv2.circle(img, center, radius, color, thickness)

def draw_radial_ticks(img, center, radius, color, num_ticks=24, length=22, thickness=3):
    # Draw radial ticks (reference style)
    for i in range(num_ticks):
        angle = np.deg2rad(i * (360/num_ticks))
        x1 = int(center[0] + (radius-length) * np.cos(angle))
        y1 = int(center[1] + (radius-length) * np.sin(angle))
        x2 = int(center[0] + radius * np.cos(angle))
        y2 = int(center[1] + radius * np.sin(angle))
        cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def draw_core_pattern(img, center, radius):
    # Draw stylized core (reference style)
    for t in np.linspace(0, 2*np.pi, 40):
        r = radius * (0.7 + 0.3 * np.sin(6*t))
        x = int(center[0] + r * np.cos(t))
        y = int(center[1] + r * np.sin(t))
        cv2.circle(img, (x, y), 3, ORANGE, -1)
    cv2.circle(img, center, int(radius*0.6), CYAN, 2)
    cv2.circle(img, center, int(radius*0.4), ORANGE, 2)

def draw_hud_details(img, center):
    # Draw bottom HUD bars and segments (reference style)
    for i in range(8):
        angle = np.deg2rad(210 + i*10)
        x1 = int(center[0] + 140 * np.cos(angle))
        y1 = int(center[1] + 140 * np.sin(angle))
        x2 = int(center[0] + 170 * np.cos(angle))
        y2 = int(center[1] + 170 * np.sin(angle))
        cv2.line(img, (x1, y1), (x2, y2), CYAN, 4)
    # Draw HUD blocks
    for i in range(4):
        angle = np.deg2rad(270 + i*15)
        x = int(center[0] + 120 * np.cos(angle))
        y = int(center[1] + 120 * np.sin(angle))
        cv2.rectangle(img, (x-10, y-10), (x+10, y+10), CYAN, 2)

def draw_arc_segments(img, center):
    # Draw arc segments (reference style)
    cv2.ellipse(img, center, (110,110), 0, -30, 210, CYAN, 3)
    cv2.ellipse(img, center, (100,100), 0, -30, 210, ORANGE, 2)
    cv2.ellipse(img, center, (80,80), 0, 0, 360, CYAN, 1)

# Start webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            h, w, _ = frame.shape
            lm = [(int(l.x * w), int(l.y * h)) for l in hand_landmarks.landmark]

            # Draw hand skeleton
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            palm = lm[9]
            tips = [lm[i] for i in [4, 8, 12, 16, 20]]
            dists = [np.linalg.norm(np.array(tip) - np.array(palm)) for tip in tips]
            avg_dist = np.mean(dists)

            # Pinch detection (thumb tip to index tip)
            pinch_dist = np.linalg.norm(np.array(lm[4]) - np.array(lm[8]))
            pinch_val = int(100 - min(pinch_dist, 100))

            # Gesture logic
            if avg_dist > 70:
                # Open hand: full AR UI
                draw_glow_circle(frame, palm, 120, CYAN, 3, glow=30)
                draw_glow_circle(frame, palm, 90, CYAN, 2, glow=20)
                draw_glow_circle(frame, palm, 60, ORANGE, 2, glow=10)
                draw_radial_ticks(frame, palm, 120, CYAN, num_ticks=24, length=22, thickness=3)
                draw_core_pattern(frame, palm, 35)
                draw_hud_details(frame, palm)
                draw_arc_segments(frame, palm)
                # Dynamic lines to fingertips
                for i in [4, 8, 12, 16, 20]:
                    cv2.line(frame, palm, lm[i], CYAN, 2)
                    cv2.circle(frame, lm[i], 12, ORANGE, -1)
                # Numeric overlay (angle between thumb and index)
                v1 = np.array(lm[4]) - np.array(palm)
                v2 = np.array(lm[8]) - np.array(palm)
                try:
                    angle = int(np.degrees(np.arccos(np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)+1e-6))))
                except:
                    angle = 0
                cv2.putText(frame, f'{angle}°', (palm[0]+40, palm[1]-40), cv2.FONT_HERSHEY_DUPLEX, 1.5, WHITE, 4)
            elif pinch_val < 60:
                # Pinch gesture: show orange arcs and value
                draw_glow_circle(frame, palm, 60, ORANGE, 3, glow=20)
                cv2.putText(frame, f'Pinch: {pinch_val}', (palm[0]-40, palm[1]-70), cv2.FONT_HERSHEY_SIMPLEX, 1, ORANGE, 3)
                for i in range(5):
                    cv2.ellipse(frame, (palm[0]+80, palm[1]), (30,30), 0, 180, 180+pinch_val+i*10, ORANGE, 2)
            else:
                # Fist: simple glowing circle
                draw_glow_circle(frame, palm, 60, CYAN, 3, glow=20)
                cv2.putText(frame, 'FIST', (palm[0]-30, palm[1]-70), cv2.FONT_HERSHEY_SIMPLEX, 1, ORANGE, 3)

    cv2.imshow('Hand Tracking AR UI', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
cap.release()
cv2.destroyAllWindows()