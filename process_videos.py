import cv2
import mediapipe as mp
import numpy as np
import os
import json
from tqdm import tqdm # Thư viện giúp tạo thanh tiến trình (pip install tqdm)

# --- 1. KHỞI TẠO MEDIAPIPE ---
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# --- 2. HÀM TRÍCH XUẤT KEYPOINTS (258 đặc trưng) ---
def extract_keypoints(results):
    # Dáng: 33 keypoints * 4 giá trị (x,y,z,visibility) = 132
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    
    # Tay Trái: 21 keypoints * 3 giá trị (x,y,z) = 63
    left_hand = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    
    # Tay Phải: 21 keypoints * 3 giá trị (x,y,z) = 63
    right_hand = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    
    # Tổng cộng: 132 + 63 + 63 = 258 đặc trưng
    return np.concatenate([pose, left_hand, right_hand])

# --- 3. ĐỊNH NGHĨA ĐƯỜNG DẪN ---
JSON_PATH = 'WLASL_v0.3.json'
VIDEO_DIR_PATH = 'videos' # Thư mục video đã giải nén
KEYPOINT_DIR_PATH = 'Data_Keypoints' # Thư mục lưu .npy

# Tạo thư mục lưu keypoints nếu chưa có
os.makedirs(KEYPOINT_DIR_PATH, exist_ok=True)

# --- 4. TẠO "BẢN ĐỒ" (MAP) TỪ JSON ---
print(f"Đang tải file JSON từ {JSON_PATH}...")
with open(JSON_PATH, 'r') as f:
    data = json.load(f)

# Tạo một dictionary (map) để tra cứu: video_id -> label (gloss)
# Ví dụ: '01234' -> 'book'
video_to_label_map = {}
for entry in data:
    label = entry['gloss'] # 'gloss' là tên của nhãn (ví dụ: 'book')
    for instance in entry['instances']:
        video_id = instance['video_id']
        video_to_label_map[video_id] = label

print(f"Đã tạo map cho {len(video_to_label_map)} video.")

# --- 5. LẶP QUA VIDEO VÀ TRÍCH XUẤT ---
print(f"Bắt đầu xử lý video từ thư mục: {VIDEO_DIR_PATH}")

# Lấy danh sách tất cả video_id có trong thư mục videos/
# (Giả sử tên file là 'video_id.mp4', ví dụ '01234.mp4')
video_files = os.listdir(VIDEO_DIR_PATH)

# Dùng tqdm để tạo thanh tiến trình đẹp mắt
for video_file in tqdm(video_files, desc="Đang xử lý video"):
    video_id = os.path.splitext(video_file)[0] # Lấy ID (ví dụ '01234')
    
    # 1. Tra cứu nhãn (label)
    if video_id not in video_to_label_map:
        print(f"Cảnh báo: Bỏ qua video {video_file} vì không tìm thấy trong JSON.")
        continue
    label = video_to_label_map[video_id]

    # 2. Tạo đường dẫn lưu file .npy
    # Lưu vào thư mục con theo tên nhãn: Data_Keypoints/book/01234.npy
    label_dir = os.path.join(KEYPOINT_DIR_PATH, label)
    os.makedirs(label_dir, exist_ok=True)
    npy_path = os.path.join(label_dir, f"{video_id}.npy")
    
    # Nếu file .npy đã tồn tại, bỏ qua
    if os.path.exists(npy_path):
        continue

    # 3. Mở video
    video_path = os.path.join(VIDEO_DIR_PATH, video_file)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Lỗi: Không thể mở video {video_path}")
        continue
        
    # 4. Lặp qua các frame và trích xuất
    sequence = [] # List để lưu các frame keypoints
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break # Hết video

        # Xử lý MediaPipe
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = holistic.process(image_rgb)
        
        # Lấy 258 đặc trưng
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)

    cap.release()
    
    # 5. Lưu chuỗi (sequence) thành file .npy
    if len(sequence) > 0:
        np.save(npy_path, np.array(sequence))

print("--- XỬ LÝ HOÀN TẤT! ---")
holistic.close()