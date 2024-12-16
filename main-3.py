import cv2
import numpy as np
import mediapipe as mp
from collections import deque

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.85, min_tracking_confidence=0.85)

# Initialize variables
drawing_color = (0, 0, 0)  # Default: Black
brush_thickness = 5
eraser_thickness = 50
eraser_mode = False
whiteboard = None
prev_x, prev_y = 0, 0
undo_stack = deque(maxlen=20)
selected_color_index = 0
line_mode = False
line_start = None
pause_drawing = False  # New variable to track drawing state
points = deque(maxlen=2)  # Store recent positions for smoothing
selected_brush_index = 0  # Initialize selected brush index

# Color Palette
color_palette = [
    (179, 179, 179),  # Muted Black
    (179, 102, 102),  # Muted Blue
    (102, 179, 102),  # Muted Green
    (102, 102, 179),  # Muted Red
    (102, 179, 179),  # Muted Yellow
    (179, 102, 179),  # Muted Magenta
    (179, 179, 102),  # Muted Cyan
]

# Brush Sizes
brush_sizes = [5, 10, 15, 20, 25]  # Different brush sizes

# Webcam capture
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Width
cap.set(4, 720)  # Height

def draw_palette(whiteboard, selected_color_index, selected_brush_index):
    """Draw the muted color palette, brush sizes, and eraser button."""
    # Color palette
    for i, color in enumerate(color_palette):
        top_left = (i * 150 + 20, 20)
        bottom_right = (i * 150 + 120, 120)
        cv2.rectangle(whiteboard, top_left, bottom_right, color, -1)
        border_color = (255, 255, 255) if i == selected_color_index and not eraser_mode else (0, 0, 0)
        cv2.rectangle(whiteboard, top_left, bottom_right, border_color, 3)

    # Brush sizes on the right, moved inwards
    for j, size in enumerate(brush_sizes):
        top_left = (whiteboard.shape[1] - 200, j * 150 + 20)
        bottom_right = (whiteboard.shape[1] - 100, j * 150 + 120)
        cv2.circle(whiteboard, ((top_left[0] + bottom_right[0]) // 2, (top_left[1] + bottom_right[1]) // 2), size, (0, 0, 0), -1)
        border_color = (255, 255, 255) if j == selected_brush_index else (0, 0, 0)
        cv2.rectangle(whiteboard, top_left, bottom_right, border_color, 3)

    # Draw the eraser button
    eraser_top_left = (len(color_palette) * 150 + 20, 20)
    eraser_bottom_right = (len(color_palette) * 150 + 120, 120)
    eraser_color = (220, 220, 220)  # Light gray for the eraser button
    cv2.rectangle(whiteboard, eraser_top_left, eraser_bottom_right, eraser_color, -1)
    text_color = (0, 0, 0) if not eraser_mode else (255, 255, 255)  # Invert text color when selected
    cv2.putText(whiteboard, "Eraser", (eraser_top_left[0] + 10, eraser_top_left[1] + 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)

    # Highlight the eraser button if selected
    if eraser_mode:
        cv2.rectangle(whiteboard, eraser_top_left, eraser_bottom_right, (255, 255, 255), 3)

    # Draw the clear all button on top right
    clear_top_left = (whiteboard.shape[1] - 1250, 570)
    clear_bottom_right = (whiteboard.shape[1] - 1100, 670)
    clear_color = (200, 200, 200)  # Light gray for the clear button
    cv2.rectangle(whiteboard, clear_top_left, clear_bottom_right, clear_color, -1)
    cv2.putText(whiteboard, "Clear", (clear_top_left[0] + 10, clear_top_left[1] + 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

def process_hand(hand_landmarks, frame, whiteboard):
    """Process hand landmarks for drawing and line mode."""
    global prev_x, prev_y, drawing_color, eraser_mode, brush_thickness, selected_color_index, line_mode, line_start, pause_drawing, points, selected_brush_index

    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    x, y = int(index_tip.x * frame.shape[1]), int(index_tip.y * frame.shape[0])

    points.append((x, y))
    avg_x = int(np.mean([pt[0] for pt in points]))
    avg_y = int(np.mean([pt[1] for pt in points]))

    # Draw the hovering pointer
    cv2.circle(frame, (avg_x, avg_y), 20, (0, 0, 255), -1)  # Red pointer on the webcam feed
    cv2.circle(frame, (avg_x, avg_y), 5, (255, 255, 255), -1)

    if line_mode:
        # Handle dynamic line drawing
        if line_start is None:
            line_start = (avg_x, avg_y)  # Set starting point
        else:
            temp_frame = whiteboard.copy()
            cv2.line(temp_frame, line_start, (avg_x, avg_y), drawing_color, brush_thickness)
            cv2.imshow("Whiteboard", cv2.addWeighted(frame, 0.4, temp_frame, 0.6, 0))
        return  # Do nothing else while in line mode

    # Check interaction with color palette
    for i, color in enumerate(color_palette):
        top_left = (i * 150 + 20, 20)
        bottom_right = (i * 150 + 120, 120)
        if top_left[0] <= avg_x <= bottom_right[0] and top_left[1] <= avg_y <= bottom_right[1]:
            drawing_color = color
            selected_color_index = i
            eraser_mode = False  # Disable eraser mode when selecting a color
            return

    # Check interaction with brush sizes
    for j, size in enumerate(brush_sizes):
        top_left = (whiteboard.shape[1] - 200, j * 150 + 20)
        bottom_right = (whiteboard.shape[1] - 100, j * 150 + 120)
        if top_left[0] <= avg_x <= bottom_right[0] and top_left[1] <= avg_y <= bottom_right[1]:
            brush_thickness = size
            selected_brush_index = j
            return

    # Check interaction with eraser button
    eraser_top_left = (len(color_palette) * 150 + 20, 20)
    eraser_bottom_right = (len(color_palette) * 150 + 120, 120)
    if eraser_top_left[0] <= avg_x <= eraser_bottom_right[0] and eraser_top_left[1] <= avg_y <= eraser_bottom_right[1]:
        eraser_mode = True
        selected_color_index = -1  # Deselect colors
        return

    # Check interaction with clear button
    clear_top_left = (whiteboard.shape[1] - 1250, 570)
    clear_bottom_right = (whiteboard.shape[1] - 1100, 670)
    if clear_top_left[0] <= avg_x <= clear_bottom_right[0] and clear_top_left[1] <= avg_y <= clear_bottom_right[1]:
        whiteboard[:] = 255  # Clear the whiteboard
        return

    # Drawing logic
    if not pause_drawing:  # Check if drawing is paused
        if eraser_mode:
            cv2.circle(whiteboard, (avg_x, avg_y), eraser_thickness, (255, 255, 255), -1)
            prev_x, prev_y = avg_x, avg_y
        else:
            if prev_x == 0 and prev_y == 0 or pause_drawing:  # Start new line segment after pause
                prev_x, prev_y = avg_x, avg_y
            else:
                undo_stack.append(whiteboard.copy())
                cv2.line(whiteboard, (prev_x, prev_y), (avg_x, avg_y), drawing_color, brush_thickness)
                prev_x, prev_y = avg_x, avg_y

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame horizontally
    frame = cv2.flip(frame, 1)

    # Convert frame to RGB for MediaPipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and detect hands
    result = hands.process(rgb_frame)

    # Initialize whiteboard if not done
    if whiteboard is None:
        whiteboard = np.ones_like(frame) * 255  # White background canvas

    # Draw the color palette and brush sizes
    draw_palette(whiteboard, selected_color_index, selected_brush_index)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            process_hand(hand_landmarks, frame, whiteboard)
    else:
        prev_x, prev_y = 0, 0  # Reset if no hand is detected

    # Blend whiteboard with the camera feed for translucency
    alpha = 0.6  # Adjust the transparency level
    translucent_frame = cv2.addWeighted(frame, 1 - alpha, whiteboard, alpha, 0)

    # Display the translucent whiteboard
    cv2.imshow("Whiteboard", translucent_frame)

    key = cv2.waitKey(1) & 0xFF

    # Exit the program
    if key == ord('q'):
        break

    # Save the whiteboard
    if key == ord('s'):
        cv2.imwrite("drawing_output.png", whiteboard)
        print("Drawing saved as drawing_output.png")

    # Undo the last action
    if key == ord('u') and len(undo_stack) > 0:
        whiteboard = undo_stack.pop()

    # Line mode toggle
    if key == ord('l'):
        if line_mode:
            # Finalize the line
            if line_start is not None:
                current_x, current_y = prev_x, prev_y
                cv2.line(whiteboard, line_start, (current_x, current_y), drawing_color, brush_thickness)
                line_start = None  # Reset start point
            line_mode = False  # Exit line mode
        else:
            line_mode = True  # Enter line mode

    # Pause drawing mode
    if key == ord(' '):
        pause_drawing = not pause_drawing
        prev_x, prev_y = 0, 0  # Reset previous coordinates when resuming

# Release the webcam and destroy all OpenCV windows
cap.release()
cv2.destroyAllWindows()
