import cv2
import numpy as np
import torch
import torch.nn as nn
import mediapipe as mp


import os
import pandas as pd

mp_holistic = mp.solutions.holistic
print("="*70)
print("🎥 VIDEO INFERENCE - TEST YOUR MODEL")
print("="*70)


# ===== LOAD MODEL =====
MODEL_PATH = "best_model.pt"  # Just the filename, not directory
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"\n✓ Loading model from: {MODEL_PATH}")
print(f"✓ File exists: {os.path.exists(MODEL_PATH)}")

if not os.path.exists(MODEL_PATH):
    print("\nERROR: Model file not found!")
    print(f"Current directory: {os.getcwd()}")
    print(f"Files in current directory: {os.listdir('.')}")
    exit()


# Model class (same as training)
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


# Load checkpoint - FIXED
checkpoint = torch.load(
    MODEL_PATH,  
    map_location=DEVICE,
    weights_only=False
)
le = checkpoint['label_encoder']
num_classes = checkpoint['num_classes']

model = AdvancedBiLSTM(225, 512, 3, num_classes, 0.5).to(DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"✓ Model loaded successfully!")
print(f"  Device: {DEVICE}")
print(f"  Classes ({num_classes}): {list(le.classes_)}")


# ===== NORMALIZATION (SAME AS TRAINING) =====
def normalize_pose(pose):
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



def predict_video(video_path, max_frames=120):
    """Extract keypoints from video and predict sign language word"""
    print(f"\n{'='*70}")
    print(f"Processing: {os.path.basename(video_path)}")
    print(f"{'='*70}")

   
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=0, 
        enable_segmentation=False,
        smooth_landmarks=True,
        min_detection_confidence=0.5, 
        min_tracking_confidence=0.5
    )


    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Cannot open video!")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video info: {total_frames} frames @ {fps:.1f} FPS")

    frames_data = []
    frame_count = 0
    shoulder_visible = 0
    hands_visible = 0

    print("Extracting keypoints...", end='', flush=True)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        

        if frame_count % 10 == 0:
            print('.', end='', flush=True)

        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = holistic.process(frame_rgb)
        frame_rgb.flags.writeable = True

     
        pose_vector = []

 
        if results.pose_landmarks:
            for lm in results.pose_landmarks.landmark:
                pose_vector.extend([lm.x, lm.y, lm.z])
            
        
            if (results.pose_landmarks.landmark[11].visibility > 0.5 and 
                results.pose_landmarks.landmark[12].visibility > 0.5):
                shoulder_visible += 1
        else:
            pose_vector.extend([0.0] * 99)

     
        if results.left_hand_landmarks:
            for lm in results.left_hand_landmarks.landmark:
                pose_vector.extend([lm.x, lm.y, lm.z])
            hands_visible += 1
        else:
            pose_vector.extend([0.0] * 63)

   
        if results.right_hand_landmarks:
            for lm in results.right_hand_landmarks.landmark:
                pose_vector.extend([lm.x, lm.y, lm.z])
            hands_visible += 1
        else:
            pose_vector.extend([0.0] * 63)

        frames_data.append(pose_vector)

    cap.release()
    holistic.close()
    print()  

    print(f"✓ Extracted {len(frames_data)} frames")
    print(f"  Shoulder visibility: {shoulder_visible/len(frames_data)*100:.0f}%")
    print(f"  Hands visibility: {hands_visible/(len(frames_data)*2)*100:.0f}%")

    if len(frames_data) == 0:
        print("❌ No keypoints extracted!")
        return None

    
    pose = np.array(frames_data, dtype=np.float32)
    
    
    zero_ratio = np.sum(pose == 0.0) / pose.size
    print(f"  Data completeness: {(1-zero_ratio)*100:.1f}%")


    try:
        pose = normalize_pose(pose)
        print(f"✓ Normalized (mean={np.abs(pose).mean():.3f})")
    except Exception as e:
        print(f"Normalization failed: {e}")
        return None

  
    T = pose.shape[0]
    if T > max_frames:
   
        indices = np.linspace(0, T - 1, max_frames).astype(int)
        pose = pose[indices]
        print(f"✓ Resampled: {T} → {max_frames} frames")
        T = max_frames

   
    if T >= max_frames:
        pose_final = pose[:max_frames]
        mask = np.ones(max_frames, dtype=np.float32)
    else:
        pad = np.zeros((max_frames - T, 225), dtype=np.float32)
        pose_final = np.concatenate([pose, pad], axis=0)
        mask = np.zeros(max_frames, dtype=np.float32)
        mask[:T] = 1.0

  
    pose_tensor = torch.from_numpy(pose_final).unsqueeze(0).to(DEVICE)
    mask_tensor = torch.from_numpy(mask).unsqueeze(0).to(DEVICE)

  
    with torch.no_grad():
        logits = model(pose_tensor, mask_tensor)
        probs = torch.softmax(logits, dim=1)[0]
        top5_probs, top5_idx = torch.topk(probs, min(5, num_classes))


    print(f"\n{'='*70}")
    print("🎯 PREDICTION RESULTS")
    print(f"{'='*70}")

    predicted_word = le.classes_[top5_idx[0].item()]
    confidence = top5_probs[0].item()

    print(f"\n✨ Predicted Sign: {predicted_word.upper()}")
    print(f"   Confidence: {confidence*100:.2f}%")

    print(f"\n📊 Top 5 Predictions:")
    for i, (prob, idx) in enumerate(zip(top5_probs, top5_idx), 1):
        word = le.classes_[idx.item()]
        bar = '█' * int(prob * 30)
        print(f"  {i}. {word:<15} {bar} {prob*100:5.2f}%")

    return {
        'predicted_word': predicted_word,
        'confidence': confidence,
        'top5': [(le.classes_[idx.item()], prob.item())
                 for prob, idx in zip(top5_probs, top5_idx)]
    }



print("\n" + "="*70)
print("🎬 VIDEO TESTING MODE")
print("="*70)

VIDEO_DIR = r"D:\archive\videos"
CSV_PATH = "top15_train_ready.csv"


if not os.path.exists(CSV_PATH):
    print(f"\n❌ CSV not found: {CSV_PATH}")
    print(f"Current directory: {os.getcwd()}")
    exit()

if not os.path.exists(VIDEO_DIR):
    print(f"\n❌ Video directory not found: {VIDEO_DIR}")
    exit()

df = pd.read_csv(CSV_PATH)
test_videos = df[df['subset'] == 'test'].reset_index(drop=True)

print(f"\n✓ Found {len(test_videos)} test videos")

correct_count = 0
total_count = 0
results_log = []

try:
    while True:
        print("\n" + "="*70)
        print(f"📊 Score: {correct_count}/{total_count} correct", end='')
        if total_count > 0:
            print(f" ({correct_count/total_count*100:.1f}%)")
        else:
            print()
        print("="*70)
        
        print(f"\nAvailable test videos ({len(test_videos)} total):")
        print(f"\n{'#':<5} {'Video ID':<12} {'True Label':<15}")
        print("-" * 40)

        # Show first 20 videos
        for i, (idx, row) in enumerate(test_videos.head(20).iterrows(), 1):
            print(f"{i:<5} {row['video_id']:<12} {row['gloss']:<15}")
        
        if len(test_videos) > 20:
            print(f"... and {len(test_videos)-20} more")

        print("\n" + "="*70)
        print("Options:")
        print(f"  1-{len(test_videos)}: Test specific video")
        print("  'r': Random video")
        print("  'all': Test ALL videos")
        print("  'q': Quit")
        print("="*70)

        choice = input("\nYour choice: ").strip()

        if choice.lower() == 'q':
            break

        elif choice.lower() == 'all':
            print(f"\n🚀 Testing ALL {len(test_videos)} videos...")
            confirm = input("This may take a while. Continue? (y/n): ")
            if confirm.lower() != 'y':
                continue
            
            for idx, test_row in test_videos.iterrows():
                video_id = str(test_row['video_id']).zfill(5)
                true_label = test_row['gloss']
                video_path = os.path.join(VIDEO_DIR, f"{video_id}.mp4")

                if os.path.exists(video_path):
                    print(f"\n[{total_count+1}/{len(test_videos)}] Testing: {video_id} (True: {true_label})")
                    result = predict_video(video_path)

                    if result:
                        total_count += 1
                        is_correct = result['predicted_word'].lower() == true_label.lower()
                        
                        if is_correct:
                            correct_count += 1
                            print("✅ CORRECT!")
                        else:
                            print(f"❌ WRONG - Expected: {true_label.upper()}")
                        
                        results_log.append({
                            'video_id': video_id,
                            'true': true_label,
                            'predicted': result['predicted_word'],
                            'confidence': result['confidence'],
                            'correct': is_correct
                        })
                else:
                    print(f"⚠️  Video not found: {video_path}")

            print(f"\n{'='*70}")
            print(f"🏆 BATCH COMPLETE")
            print(f"   Accuracy: {correct_count}/{total_count} = {correct_count/total_count*100:.1f}%")
            print(f"{'='*70}")
            
           
            if results_log:
                results_df = pd.DataFrame(results_log)
                results_df.to_csv('test_results.csv', index=False)
                print(f"✓ Results saved to test_results.csv")
            
            input("\nPress Enter to continue...")
            continue

        elif choice.lower() == 'r':
            test_row = test_videos.sample(1).iloc[0]
            print(f"\n🎲 Random: {test_row['video_id']}")

        else:
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(test_videos):
                    test_row = test_videos.iloc[idx]
                else:
                    print("❌ Invalid number!")
                    continue
            except:
                print("❌ Invalid input!")
                continue

   
        video_id = str(test_row['video_id']).zfill(5)
        true_label = test_row['gloss']
        video_path = os.path.join(VIDEO_DIR, f"{video_id}.mp4")

        if os.path.exists(video_path):
            print(f"\n🎥 Testing: {video_id}")
            print(f"   True label: {true_label.upper()}")

            result = predict_video(video_path)

            if result:
                total_count += 1
                is_correct = result['predicted_word'].lower() == true_label.lower()
                
                print(f"\n{'='*70}")
                if is_correct:
                    correct_count += 1
                    print("✅ CORRECT PREDICTION!")
                else:
                    print(f"❌ WRONG - Expected: {true_label.upper()}, Got: {result['predicted_word'].upper()}")
                print(f"{'='*70}")
                
                results_log.append({
                    'video_id': video_id,
                    'true': true_label,
                    'predicted': result['predicted_word'],
                    'confidence': result['confidence'],
                    'correct': is_correct
                })
        else:
            print(f"\n❌ Video not found: {video_path}")

        input("\nPress Enter to continue...")

except KeyboardInterrupt:
    print("\n\nInterrupted by user")

print("\n" + "="*70)
print("🏁 TESTING COMPLETE")
print("="*70)
if total_count > 0:
    print(f"Final Score: {correct_count}/{total_count} = {correct_count/total_count*100:.1f}%")
    
    if results_log:
        results_df = pd.DataFrame(results_log)
        results_df.to_csv('test_results.csv', index=False)
        print(f"\n✓ Detailed results saved to test_results.csv")
        
        
        print("\n📊 Per-class accuracy:")
        for word in sorted(results_df['true'].unique()):
            word_results = results_df[results_df['true'] == word]
            word_correct = word_results['correct'].sum()
            word_total = len(word_results)
            print(f"  {word:<15} {word_correct}/{word_total} ({word_correct/word_total*100:.0f}%)")
else:
    print("No videos tested")

print("\nGoodbye!")
