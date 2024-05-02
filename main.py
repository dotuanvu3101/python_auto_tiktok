import cv2
import mediapipe as mp
import pyautogui

# Khởi tạo MediaPipe Hands
mp_hands = mp.solutions.hands
drawing_utils = mp.solutions.drawing_utils

# Khởi tạo PyAutoGUI
pyautogui.FAILSAFE = False

# Khởi tạo webcam
cap = cv2.VideoCapture(0)

while True:
    # Đọc ảnh từ webcam
    ret, frame = cap.read()

    # Chuyển đổi ảnh sang RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Xử lý ảnh bằng MediaPipe Hands
    with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        # Tìm các điểm landmark trên bàn tay
        results = hands.process(rgb_frame)

        # Kiểm tra xem có tìm thấy tay nào hay không
        if results.multi_hand_landmarks:
            # Lấy điểm landmark của ngón trỏ
            index_finger_tip = results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            # Tính toán tọa độ x và y của ngón adwdq
            x = index_finger_tip.x * frame.shape[1]
            y = index_finger_tip.y * frame.shape[0]

            # Kiểm soát cử chỉ tay
            if y < 0.30 * frame.shape[0]:  # Vuốt lên (chuyển sang video tiếp theo)
                pyautogui.scroll(-100)

            # Vẽ các điểm landmark trên bàn tay
            for hand_landmarks in results.multi_hand_landmarks:
                drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Hiển thị ảnh
    cv2.imshow('tool tiktok', frame)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng webcam
cap.release()

# Đóng cửa sổ
cv2.destroyAllWindows()
