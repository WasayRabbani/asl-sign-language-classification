
"""Training_BILSTM.ipynb




from google.colab import drive
drive.mount('/content/drive')

"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import json

print("="*70)
print("🔥 ULTIMATE BiLSTM TRAINING - MAXIMUM ACCURACY 🔥")
print("="*70)

# ===== PATHS =====
NPY_DIR = "/content/drive/.shortcut-targets-by-id/1LggtEMRg-c98uUfa21wlupsl_WT35lxD/FYP_word/npy_arrays"
CSV_PATH = "/content/drive/MyDrive/FYP_word/top15_train_ready.csv"
SAVE_DIR = "/content/drive/MyDrive/FYP_word/best_bilstm_model"
os.makedirs(SAVE_DIR, exist_ok=True)

# ===== HYPERPARAMETERS (OPTIMIZED) =====
MAX_FRAMES = 120
HIDDEN_DIM = 512      # Larger for better capacity
NUM_LAYERS = 3        # Deeper network
BATCH_SIZE = 16
LEARNING_RATE = 5e-4  # Higher initial LR with scheduler
NUM_EPOCHS = 30
DROPOUT = 0.5         # Strong regularization
PATIENCE = 7          # Early stopping patience
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"\n✓ Using device: {DEVICE}")
print(f"✓ Hyperparameters:")
print(f"  Hidden dim: {HIDDEN_DIM}")
print(f"  Layers: {NUM_LAYERS}")
print(f"  Dropout: {DROPOUT}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Max frames: {MAX_FRAMES}")

# ===== LOAD DATA =====
df = pd.read_csv(CSV_PATH)
print(f"\n✓ Loaded CSV: {len(df)} samples")
print(f"  Split distribution:\n{df['subset'].value_counts()}")

# Encode labels
le = LabelEncoder()
df['label'] = le.fit_transform(df['gloss'])
num_classes = len(le.classes_)

print(f"\n✓ Classes: {num_classes}")
print(f"  Words: {list(le.classes_)}")

# Check for missing .npy files
print(f"\n✓ Checking .npy files...")
existing_mask = []
missing_count = 0

for idx, row in df.iterrows():
    video_id = str(row['video_id']).zfill(5)
    npy_path = os.path.join(NPY_DIR, f"{video_id}.npy")
    exists = os.path.exists(npy_path)
    existing_mask.append(exists)
    if not exists:
        missing_count += 1

df = df[existing_mask].reset_index(drop=True)
print(f"  Available: {len(df)} samples")
print(f"  Missing: {missing_count} files")

# Re-encode after filtering
df['label'] = le.fit_transform(df['gloss'])
num_classes = len(le.classes_)

# ===== ADVANCED NORMALIZATION =====
def normalize_pose(pose):
    """
    Advanced normalization:
    1. Center on shoulder midpoint
    2. Scale by shoulder width
    3. Remove outliers
    """
    if pose.shape[0] == 0:
        return pose

    pose_reshaped = pose.reshape(pose.shape[0], -1, 3)  # (T, 75, 3)

    # MediaPipe landmarks: 11=left_shoulder, 12=right_shoulder
    left_shoulder = pose_reshaped[:, 11, :]
    right_shoulder = pose_reshaped[:, 12, :]

    # Center on shoulder midpoint
    shoulder_mid = (left_shoulder + right_shoulder) / 2.0
    shoulder_width = np.linalg.norm(right_shoulder - left_shoulder, axis=1, keepdims=True)
    shoulder_width = np.maximum(shoulder_width, 0.01)  # Avoid division by zero

    # Center and scale
    pose_centered = pose_reshaped - shoulder_mid[:, None, :]
    pose_normalized = pose_centered / shoulder_width[:, :, None]

    # Clip outliers (removes noise)
    pose_normalized = np.clip(pose_normalized, -10, 10)

    return pose_normalized.reshape(pose.shape[0], -1)


# ===== DATASET WITH AUGMENTATION =====
class PoseSignDataset(Dataset):
    def __init__(self, dataframe, npy_dir, max_frames=120, normalize=True, augment=False, subset=None):
        if subset:
            self.df = dataframe[dataframe['subset'] == subset].reset_index(drop=True)
        else:
            self.df = dataframe
        self.npy_dir = npy_dir
        self.max_frames = max_frames
        self.normalize = normalize
        self.augment = augment
        print(f"  {subset if subset else 'Full'} set: {len(self.df)} samples (augment={augment})")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
      row = self.df.iloc[idx]
      video_id = str(row['video_id']).zfill(5)
      label = int(row['label'])

      npy_path = os.path.join(self.npy_dir, f"{video_id}.npy")
      pose = np.load(npy_path).astype(np.float32)  # Force float32

      # Normalize
      if self.normalize:
          pose = normalize_pose(pose)
          pose = pose.astype(np.float32)  # Ensure float32 after normalization

      # Data augmentation (training only)
      if self.augment:
          # Random temporal cropping
          T = pose.shape[0]
          if T > self.max_frames:
              max_start = T - self.max_frames
              start = np.random.randint(0, max_start + 1)
              pose = pose[start:start + self.max_frames]

          # Random noise injection (small)
          if np.random.random() > 0.5:
              noise = np.random.normal(0, 0.02, pose.shape).astype(np.float32)
              pose = pose + noise

          # Random temporal scaling (speed variation)
          if np.random.random() > 0.5:
              scale = np.random.uniform(0.9, 1.1)
              new_len = int(pose.shape[0] * scale)
              if new_len > 5:
                  indices = np.linspace(0, pose.shape[0]-1, new_len).astype(int)
                  pose = pose[indices]

      # Ensure float32 before padding
      pose = pose.astype(np.float32)

      # Pad/truncate to fixed length
      T = pose.shape[0]
      if T >= self.max_frames:
          pose = pose[:self.max_frames]
          mask = np.ones(self.max_frames, dtype=np.float32)
      else:
          pad = np.zeros((self.max_frames - T, 225), dtype=np.float32)
          pose = np.concatenate([pose, pad], axis=0)
          mask = np.zeros(self.max_frames, dtype=np.float32)
          mask[:T] = 1.0

      pose = torch.from_numpy(pose.astype(np.float32))  # Explicit float32
      mask = torch.from_numpy(mask.astype(np.float32))
      label = torch.tensor(label, dtype=torch.long)

      return pose, mask, label



# ===== CREATE DATASETS =====
print(f"\n✓ Creating datasets...")
train_dataset = PoseSignDataset(df, NPY_DIR, MAX_FRAMES, normalize=True, augment=True, subset='train')
val_dataset = PoseSignDataset(df, NPY_DIR, MAX_FRAMES, normalize=True, augment=False, subset='val')
test_dataset = PoseSignDataset(df, NPY_DIR, MAX_FRAMES, normalize=True, augment=False, subset='test')

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)


# ===== ADVANCED BiLSTM MODEL =====
class AdvancedBiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, dropout=0.5):
        super().__init__()

        # Batch normalization for input
        self.bn_input = nn.BatchNorm1d(input_dim)

        # BiLSTM
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.bn_fc = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, mask):
        # Input batch norm
        x = self.bn_input(x.transpose(1, 2)).transpose(1, 2)

        # BiLSTM
        lstm_out, _ = self.lstm(x)

        # Attention weights
        attention_scores = self.attention(lstm_out).squeeze(-1)
        attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        attention_weights = torch.softmax(attention_scores, dim=1)

        # Weighted sum
        context = torch.sum(lstm_out * attention_weights.unsqueeze(-1), dim=1)

        # Classification
        out = self.dropout(context)
        out = self.fc1(out)
        out = self.bn_fc(out)
        out = self.relu(out)
        out = self.dropout(out)
        logits = self.fc2(out)

        return logits


model = AdvancedBiLSTM(225, HIDDEN_DIM, NUM_LAYERS, num_classes, DROPOUT).to(DEVICE)
print(f"\n✓ Model created:")
print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=3
)

best_val_acc = 0.0
patience_counter = 0
train_losses = []
val_losses = []
train_accs = []
val_accs = []



def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    total_loss = 0.0

    with torch.no_grad():
        for poses, masks, labels in loader:
            poses, masks, labels = poses.to(device), masks.to(device), labels.to(device)
            logits = model(poses, masks)
            loss = criterion(logits, labels)

            total_loss += loss.item() * poses.size(0)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    top1_acc = (all_preds == all_labels).mean()
    top5_preds = np.argsort(all_probs, axis=1)[:, -5:]
    top5_acc = np.mean([label in top5_preds[i] for i, label in enumerate(all_labels)])
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    avg_loss = total_loss / len(all_labels)

    return avg_loss, top1_acc, top5_acc, f1, all_preds, all_labels



print("\n" + "="*70)
print("🚀 TRAINING START")
print("="*70)

for epoch in range(1, NUM_EPOCHS + 1):
  
    model.train()
    train_loss, train_correct, train_total = 0.0, 0, 0

    for poses, masks, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [Train]", leave=False):
        poses, masks, labels = poses.to(DEVICE), masks.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        logits = model(poses, masks)
        loss = criterion(logits, labels)
        loss.backward()

   
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        train_loss += loss.item() * poses.size(0)
        preds = logits.argmax(dim=1)
        train_correct += (preds == labels).sum().item()
        train_total += labels.size(0)

    train_loss /= train_total
    train_acc = train_correct / train_total


    val_loss, val_top1, val_top5, val_f1, _, _ = evaluate(model, val_loader, DEVICE)


    scheduler.step(val_top1)
    current_lr = optimizer.param_groups[0]['lr']

   
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_top1)

    print(f"\nEpoch {epoch}/{NUM_EPOCHS}:")
    print(f"  Train: loss={train_loss:.4f}, acc={train_acc:.4f}")
    print(f"  Val:   loss={val_loss:.4f}, top1={val_top1:.4f}, top5={val_top5:.4f}, f1={val_f1:.4f}, lr={current_lr:.6f}")


    if val_top1 > best_val_acc:
        best_val_acc = val_top1
        patience_counter = 0

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_top1': val_top1,
            'val_top5': val_top5,
            'val_f1': val_f1,
            'label_encoder': le,
            'num_classes': num_classes,
            'hyperparameters': {
                'hidden_dim': HIDDEN_DIM,
                'num_layers': NUM_LAYERS,
                'dropout': DROPOUT,
                'max_frames': MAX_FRAMES
            }
        }, os.path.join(SAVE_DIR, 'best_model.pt'))
        print(f"NEW BEST MODEL SAVED!")
    else:
        patience_counter += 1


    if patience_counter >= PATIENCE:
        print(f"\nEarly stopping triggered after {epoch} epochs")
        break

print("\n" + "="*70)
print(f"✅ TRAINING COMPLETE - Best Val Acc: {best_val_acc:.4f}")
print("="*70)



print("\n" + "="*70)
print("📊 FINAL TEST SET EVALUATION")
print("="*70)

checkpoint = torch.load(os.path.join(SAVE_DIR, 'best_model.pt'), weights_only=False) 
model.load_state_dict(checkpoint['model_state_dict'])

test_loss, test_top1, test_top5, test_f1, test_preds, test_labels = evaluate(model, test_loader, DEVICE)

print(f"\n🎯 FINAL RESULTS:")
print(f"  Best Val Accuracy: {best_val_acc*100:.2f}%")
print(f"  Test Loss: {test_loss:.4f}")
print(f"  Test Top-1 Accuracy: {test_top1*100:.2f}%")
print(f"  Test Top-5 Accuracy: {test_top5*100:.2f}%")
print(f"  Test F1-Score: {test_f1:.4f}")



print("\n" + "="*70)
print("PER-CLASS PERFORMANCE")
print("="*70)

present_classes = np.unique(test_labels)
present_names = [le.classes_[i] for i in present_classes]
print(classification_report(test_labels, test_preds, labels=present_classes, target_names=present_names, digits=3))



cm = confusion_matrix(test_labels, test_preds)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title(f'Confusion Matrix - Test Accuracy: {test_top1*100:.2f}%')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'confusion_matrix.png'), dpi=300)
print(f"\n✓ Confusion matrix saved")



fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

ax1.plot(train_losses, label='Train Loss')
ax1.plot(val_losses, label='Val Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training and Validation Loss')
ax1.legend()
ax1.grid(True)

ax2.plot(train_accs, label='Train Acc')
ax2.plot(val_accs, label='Val Acc')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.set_title('Training and Validation Accuracy')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'training_curves.png'), dpi=300)
print(f"✓ Training curves saved")



results = {
    'final_train_accuracy': float(train_acc), 
    'best_val_accuracy': float(best_val_acc),
    'test_accuracy': float(test_top1),
    'test_top5_accuracy': float(test_top5),
    'test_f1_score': float(test_f1),
    'num_classes': num_classes,
    'num_train_samples': len(train_dataset),
    'num_val_samples': len(val_dataset),
    'num_test_samples': len(test_dataset),
    'classes': list(le.classes_),
    'hyperparameters': checkpoint['hyperparameters']
}

with open(os.path.join(SAVE_DIR, 'results.json'), 'w') as f:
    json.dump(results, indent=2, fp=f)

print(f"✓ Results saved to: {SAVE_DIR}/results.json")

print("\n" + "="*70)
print("ALL DONE! MODEL TRAINED AND SAVED!")
print("="*70)
print(f"\nFiles saved in: {SAVE_DIR}")
print(f"  - best_model.pt (trained model)")
print(f"  - confusion_matrix.png")
print(f"  - training_curves.png")
print(f"  - results.json")

print(f"\nSUMMARY:")
print(f"  Classes: {num_classes} words")
print(f"  Training samples: {len(train_dataset)}")

print(f"  Training accuracy: {train_acc*100:.2f}%") 
print(f"  Test accuracy: {test_top1*100:.2f}%")
print(f"  Top-5 accuracy: {test_top5*100:.2f}%")

"""# Testing model with a video"""

