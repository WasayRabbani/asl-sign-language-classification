import cv2
import numpy as np
import torch
import torch.nn as nn
from mediapipe.python.solutions import holistic as mp_holistic
from mediapipe.python.solutions import drawing_utils as mp_draw
from collections import deque
import time



print("="*70)
print("🎥 REAL-TIME SIGN LANGUAGE RECOGNITION - WAIST-UP FRAMING")
print("="*70)



MODEL_PATH = "best_model.pt"
DEVICE = torch.device('cpu')
print(f"\n✓ Using: CPU")


class AdvancedBiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, dropout=0.5):
        super().__init__()
        self.bn_input = nn.BatchNorm1d(input_dim)
        self.lstm = nn.LSTM(
            input_dim, hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.bn_fc = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, mask):
        x = self.bn_input(x.transpose(1, 2)).transpose(1, 2)
        lstm_out, _ = self.lstm(x)
        attention_scores = self.attention(lstm_out).squeeze(-1)
        attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        attention_weights = torch.softmax(attention_scores, dim=1)
        context = torch.sum(lstm_out * attention_weights.unsqueeze(-1), dim=1)
        out = self.dropout(context)
        out = self.fc1(out)
        out = self.bn_fc(out)
        out = self.relu(out)
        out = self.dropout(out)
        logits = self.fc2(out)
        return logits



checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
le = checkpoint['label_encoder']
num_classes = checkpoint['num_classes']

model = AdvancedBiLSTM(225, 512, 3, num_classes, 0.5).to(DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"✓ Model loaded!")
print(f"  Words ({num_classes}): {list(le.classes_)}")


# ===== NORMALIZATION =====
def normalize_pose(pose):
    """EXACT same normalization as training"""
    if pose.shape[0] == 0:
        return pose
    
    pose_reshaped = pose.reshape(pose.shape[0], -1, 3)
    
    left_shoulder = pose_reshaped[:, 11, :]
    right_shoulder = pose_reshaped[:, 12, :]
    
    shoulder_mid = (left_shoulder + right_shoulder) / 2.0
    shoulder_width = np.linalg.norm(right_shoulder - left_shoulder, axis=1, keepdims=True)
    shoulder_width = np.maximum(shoulder_width, 0.01)
    
    pose_centered = pose_reshaped - shoulder_mid[:, None, :]
    pose_normalized = pose_centered / shoulder_width[:, :, None]
    pose_normalized = np.clip(pose_normalized, -10, 10)
    
    return pose_normalized.reshape(pose.shape[0], -1).astype(np.float32)


# ===== SIMPLE RECOGNIZER =====
class RealtimeSignRecognizer:
    def __init__(self, model, label_encoder, device):
        self.model = model
        self.le = label_encoder
        self.device = device
        
        self.is_recording = False
        self.recording_start_time = None
        self.frame_buffer = deque(maxlen=150)
        
        self.current_prediction = None
        self.confidence = 0.0
        self.top3 = []
        
        self.shoulder_visible_count = 0
        self.hands_visible_count = 0
        
        # MediaPipe
        self.holistic = mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=0,
            enable_segmentation=False,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        self.prediction_history = deque(maxlen=3)
    
    def extract_keypoints(self, results):
        """Extract keypoints - FORCE FLOAT32"""
        keypoints = []
        
        # Pose
        if results.pose_landmarks:
            for lm in results.pose_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])
            
            left_shoulder = results.pose_landmarks.landmark[11]
            right_shoulder = results.pose_landmarks.landmark[12]
            if left_shoulder.visibility > 0.5 and right_shoulder.visibility > 0.5:
                self.shoulder_visible_count += 1
        else:
            keypoints.extend([0.0] * 99)
        
        # Left hand
        if results.left_hand_landmarks:
            for lm in results.left_hand_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])
            self.hands_visible_count += 1
        else:
            keypoints.extend([0.0] * 63)
        
        # Right hand
        if results.right_hand_landmarks:
            for lm in results.right_hand_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])
            self.hands_visible_count += 1
        else:
            keypoints.extend([0.0] * 63)
        
        return np.array(keypoints, dtype=np.float32)
    
    def resample_to_120_frames(self, frames):
        """Resample to exactly 120 frames"""
        current_len = len(frames)
        
        if current_len == 120:
            return frames
        elif current_len > 120:
            indices = np.linspace(0, current_len - 1, 120).astype(int)
            return [frames[i] for i in indices]
        else:
            return frames
    
    def predict(self):
        """Predict sign"""
        MIN_FRAMES = 90
        
        if len(self.frame_buffer) < MIN_FRAMES:
            print(f"\n⚠️  Need {MIN_FRAMES}+ frames (you have {len(self.frame_buffer)})")
            print(f"   Record for 3-4 seconds!")
            return
        
        print(f"\n{'='*70}")
        print(f"🔮 PREDICTING from {len(self.frame_buffer)} frames...")
        
        # Resample
        pose_list = list(self.frame_buffer)
        original_len = len(pose_list)
        
        if original_len >= 120:
            pose_list = self.resample_to_120_frames(pose_list)
            print(f"✓ Resampled: {original_len} → 120 frames")
        
        pose = np.array(pose_list, dtype=np.float32)
        
        # Quality check
        zero_ratio = np.sum(pose == 0.0) / pose.size
        print(f"✓ Data: {(1-zero_ratio)*100:.1f}% complete")
        
        if zero_ratio > 0.5:
            print(f"Too much missing data ({zero_ratio*100:.0f}%)")
            return
        
        # Normalize
        try:
            pose_normalized = normalize_pose(pose)
            print(f"✓ Normalized")
        except Exception as e:
            print(f"Normalization failed: {e}")
            return
        
        if np.isnan(pose_normalized).any() or np.isinf(pose_normalized).any():
            print("Invalid values")
            return
        
        # Pad to 120
        T = pose_normalized.shape[0]
        max_frames = 120
        
        if T >= max_frames:
            pose_final = pose_normalized[:max_frames]
            mask = np.ones(max_frames, dtype=np.float32)
        else:
            pad = np.zeros((max_frames - T, 225), dtype=np.float32)
            pose_final = np.concatenate([pose_normalized, pad], axis=0)
            mask = np.zeros(max_frames, dtype=np.float32)
            mask[:T] = 1.0
        
        # Predict
        pose_tensor = torch.from_numpy(pose_final).unsqueeze(0).to(self.device)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.model(pose_tensor, mask_tensor)
            probs = torch.softmax(logits, dim=1)[0]
            top5_probs, top5_idx = torch.topk(probs, min(5, len(self.le.classes_)))
        
        pred_idx = top5_idx[0].item()
        pred_word = self.le.classes_[pred_idx]
        pred_conf = top5_probs[0].item()
        
        self.prediction_history.append((pred_word, pred_conf))
        
        if len(self.prediction_history) >= 2:
            best_recent = max(self.prediction_history, key=lambda x: x[1])
            self.current_prediction = best_recent[0]
            self.confidence = best_recent[1]
        else:
            self.current_prediction = pred_word
            self.confidence = pred_conf
        
        self.top3 = [(self.le.classes_[idx.item()], prob.item()) 
                     for prob, idx in zip(top5_probs[:3], top5_idx[:3])]
        
        print(f"\n✨ PREDICTION: {self.current_prediction.upper()}")
        print(f"   Confidence: {self.confidence*100:.1f}%")
        print(f"\nTop 5:")
        for i, (prob, idx) in enumerate(zip(top5_probs, top5_idx), 1):
            word = self.le.classes_[idx.item()]
            bar = '█' * int(prob * 30)
            print(f"  {i}. {word:<15} {bar} {prob*100:5.1f}%")
        
        print(f"{'='*70}\n")
        
        self.shoulder_visible_count = 0
        self.hands_visible_count = 0
    
    def draw_landmarks(self, frame, results):
        """Draw landmarks"""
        if results.pose_landmarks:
            mp_draw.draw_landmarks(
                frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                mp_draw.DrawingSpec(color=(0,255,0), thickness=2)
            )
        
        if results.left_hand_landmarks:
            mp_draw.draw_landmarks(
                frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(255,100,0), thickness=2, circle_radius=2),
                mp_draw.DrawingSpec(color=(255,100,0), thickness=2)
            )
        
        if results.right_hand_landmarks:
            mp_draw.draw_landmarks(
                frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0,100,255), thickness=2, circle_radius=2),
                mp_draw.DrawingSpec(color=(0,100,255), thickness=2)
            )
    
    def draw_framing_guide(self, frame):
        """Draw guide for waist-up framing"""
        h, w, _ = frame.shape
        
       
        guide_top = int(h * 0.05)
        guide_bottom = int(h * 0.95)
        guide_left = int(w * 0.25)
        guide_right = int(w * 0.75)
        
        # Draw dotted rectangle guide
        cv2.rectangle(frame, (guide_left, guide_top), (guide_right, guide_bottom), 
                     (0, 255, 255), 2)
        
        # Labels
        cv2.putText(frame, "Position yourself here", (guide_left + 10, guide_top + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, "(Head to waist)", (guide_left + 10, guide_top + 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    def draw_ui(self, frame):
        """Draw UI"""
        h, w, _ = frame.shape
        
        # Top bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 100), (0, 0, 0), -1)
        cv2.rectangle(overlay, (0, h-200), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
        
        # Title
        cv2.putText(frame, "SIGN LANGUAGE RECOGNITION", 
                    (20, 35), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,255,255), 2)
        
        # FPS
        cv2.putText(frame, f"FPS: {self.fps:.0f}", 
                    (w-120, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        
        # Recording status
        if self.is_recording:
            elapsed = time.time() - self.recording_start_time
            frames = len(self.frame_buffer)
            
            cv2.circle(frame, (25, 70), 12, (0,0,255), -1)
            
            # Progress bar
            progress = min(frames / 120, 1.0)
            bar_w = int(300 * progress)
            cv2.rectangle(frame, (50, 60), (50+bar_w, 80), (0,255,0), -1)
            cv2.rectangle(frame, (50, 60), (350, 80), (255,255,255), 2)
            
            status = f"REC: {frames}/120 ({elapsed:.1f}s)"
            cv2.putText(frame, status, (360, 75), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            
            if elapsed >= 3.0:
                cv2.putText(frame, "Press SPACE to stop", (w-250, 75), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        else:
            cv2.circle(frame, (25, 70), 12, (128,128,128), -1)
            cv2.putText(frame, "Press SPACE to start recording", 
                       (50, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)
        
        # Prediction
        if self.current_prediction:
            cv2.putText(frame, self.current_prediction.upper(), 
                       (20, h-135), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0,255,0), 3)
            
            # Confidence bar
            bar_w = int(400 * self.confidence)
            cv2.rectangle(frame, (20, h-95), (20+bar_w, h-75), (0,255,0), -1)
            cv2.rectangle(frame, (20, h-95), (420, h-75), (255,255,255), 2)
            
            cv2.putText(frame, f"{self.confidence*100:.0f}%", 
                       (440, h-80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            
            # Top 3
            y = h-60
            for i, (word, prob) in enumerate(self.top3):
                text = f"{i+1}. {word:<12} {prob*100:4.0f}%"
                cv2.putText(frame, text, (20, y + i*20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220,220,220), 1)
        
        # Instructions
        cv2.putText(frame, "SPACE: Record (3-4s) | ENTER: Predict | C: Clear | Q: Quit", 
                    (20, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,180,180), 1)
    
    def run(self):
        """Main loop"""
        print("\n" + "="*70)
        print("STARTING WEBCAM...")
        print("="*70)
        
        # Try different camera indices
        for cam_id in [0, 1, 2]:
            cap = cv2.VideoCapture(1)
            if cap.isOpened():
                print(f"✓ Using camera {cam_id}")
                break
        
        if not cap.isOpened():
            print("Cannot open any webcam!")
            return
        
        # Resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("✓ Webcam ready!")
        print("\n📋 SETUP (MATCH YOUR TRAINING VIDEOS):")
        print("  • Position yourself HEAD TO WAIST in frame")
        print("  • Same framing as your training videos (above navel)")
        print("  • Stand 1-1.5 meters from camera")
        print("  • Good lighting")
        print("  • Record for 3-4 seconds per sign")
        print("\nControls:")
        print("  SPACE - Start/Stop recording")
        print("  ENTER - Predict")
        print("  C - Clear")
        print("  Q - Quit")
        print("="*70 + "\n")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            
            # Draw framing guide
            self.draw_framing_guide(frame)
            
            # MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False
            results = self.holistic.process(frame_rgb)
            frame_rgb.flags.writeable = True
            
            # Extract
            keypoints = self.extract_keypoints(results)
            
            if self.is_recording:
                self.frame_buffer.append(keypoints)
            
            # Draw
            self.draw_landmarks(frame, results)
            self.draw_ui(frame)
            
            # FPS
            self.frame_count += 1
            if self.frame_count % 30 == 0:
                elapsed = time.time() - self.start_time
                self.fps = 30 / max(elapsed, 0.001)
                self.start_time = time.time()
            
            cv2.imshow('Sign Language Recognition', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord(' '):
                self.is_recording = not self.is_recording
                if self.is_recording:
                    self.frame_buffer.clear()
                    self.recording_start_time = time.time()
                    self.shoulder_visible_count = 0
                    self.hands_visible_count = 0
                    print("\nRecording... perform sign for 3-4 seconds")
                else:
                    print(f"⏹️  Stopped: {len(self.frame_buffer)} frames ({time.time() - self.recording_start_time:.1f}s)")
                    print("   Press ENTER to predict")
            elif key == 13:
                self.predict()
            elif key == ord('c'):
                self.frame_buffer.clear()
                self.current_prediction = None
                self.prediction_history.clear()
                print("🗑️  Cleared")
        
        cap.release()
        cv2.destroyAllWindows()
        self.holistic.close()


# ===== RUN =====
print("\n💡 KEY POINT:")
print("   Frame yourself the SAME WAY as your training videos!")
print("   (Head to waist, above navel)\n")

recognizer = RealtimeSignRecognizer(model, le, DEVICE)
recognizer.run()


