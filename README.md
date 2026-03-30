# Track & Field Biomechanical Injury Risk Analyser

A deep learning system that analyses athlete videos to detect track & field 
events and identify injury-prone movement patterns.

## What it does
- Detects 7 track & field events from video using VideoMAE transformer
- Extracts 33 skeletal keypoints using MediaPipe Pose
- Calculates joint angles (knee, hip, elbow, shoulder, trunk, spine)
- Flags biomechanical risk patterns linked to specific injuries
- Generates a full injury risk report with annotated frame

## Events Detected
Long Jump, High Jump, Pole Vault, Shot Put, Discus Throw, Javelin Throw, Sprint

## Tech Stack
- Python, PyTorch, HuggingFace Transformers
- VideoMAE (pre-trained on Kinetics-400, fine-tuned on UCF-101)
- MediaPipe Pose
- OpenCV, NumPy, Matplotlib

## Dataset
UCF-101 Action Recognition Dataset — 7 track & field classes, ~884 clips

## Model Architecture
- Branch 1: VideoMAE backbone → 768-d video features
- Branch 2: MediaPipe pose MLP → 128-d keypoint features  
- Fusion: Concatenate → Dense(512) → Dense(256)
- Output 1: Event classification (7 classes)
- Output 2: Biomechanical injury risk analysis

## Results
- Two-phase training: head-only (5 epochs) + full fine-tune (10 epochs)
- Sprint biomechanics override for improved sprint detection
- Discus vs Shot Put disambiguation via lateral rotation analysis

## How to Run
Open the notebook in Kaggle or Google Colab with GPU enabled.

## Images 
- Sample frames Submission 
<img width="3135" height="501" alt="sample_frames_submission" src="https://github.com/user-attachments/assets/7f4f934b-55cf-454f-b53a-dea7dc35fae8" />

- Injury risk report 
<img width="2666" height="2307" alt="injury_risk_report" src="https://github.com/user-attachments/assets/0ebf8497-b03a-4ec7-a6cd-ae6603f64425" />

- Event classification 
<img width="1287" height="634" alt="event_classification" src="https://github.com/user-attachments/assets/76c1f913-af31-4d14-ab07-57dc8ba37e0e" />

- Confusion Matrix
<img width="1350" height="1050" alt="confusion_matrix" src="https://github.com/user-attachments/assets/b4744469-22cd-46f9-bd56-1bdf9cec8c34" />

- Training curves
<img width="2400" height="750" alt="training_curves" src="https://github.com/user-attachments/assets/267327aa-a99a-4e0c-b3b3-fa33fc1cc6ec" />
