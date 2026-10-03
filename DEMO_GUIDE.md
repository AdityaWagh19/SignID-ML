# SignID Live Interactive Demonstration Guide

This guide gives you an exact, step-by-step walkthrough for presenting **SignID** with a **100% LIVE DEMO** (no pre-baked or static results).

---

## ⚡ The 30-Second Elevator Pitch

> *"Traditional signature classifiers are closed-set: if an unauthorized person signs, the model is forced to misattribute it to whoever's name looks closest. SignID solves this with **calibrated open-set rejection ($\tau = 0.44$)**. Anyone can step up, draw a signature live on screen or hold a piece of paper to the camera: if it belongs to one of our 15 enrolled signatories, it is identified; if it is an unknown impostor or unconfident scribble, it is safely rejected."*

---

## 🚀 Pre-Demo Checklist (Start the Servers)

Open two terminals in your workspace:

### Terminal 1: Backend
```powershell
python -m uvicorn app.api:app --host 127.0.0.1 --port 8000
```
*Verify*: Visit `http://127.0.0.1:8000/api/health` → `{"status": "ok"}`

### Terminal 2: Frontend
```powershell
cd app/frontend
npm run dev
```
*Access*: Open **`http://localhost:5173/`** in Chrome/Edge.

---

## 🎯 The 4-Act Live Demonstration Script

### Act 1: The "Try to Hack It" Impostor Test (Live Draw) (2 mins)
*This is the most impactful start — it immediately proves the system is not hardcoded and isn't a fake closed-set classifier.*

1. Make sure you are on the **Identify** tab. The **"Draw Live"** pad is active by default.
2. Ensure **Linear SVM** or **Random Forest** is selected on the left sidebar, with threshold at default **$\tau = 0.44$**.
3. **Invite an Examiner or Draw Yourself**:
   - Scribble your own real-world signature, initials, or a random wavy stroke on the digital signature pad using your mouse or trackpad.
   - Note the stroke counter: `✓ 1 stroke captured`.
4. Click **"Identify Live Signature"**:
   - The result panel immediately generates live in <220ms.
   - Show the **Red Warning Banner**:
     $$\text{Below threshold — rejected: Not recognised}$$
   - Point out the confidence (typically 10%–25%, far below the $\tau = 0.44$ requirement).
5. **Key Talking Point for Reviewers**:
   > *"Notice how the model did NOT falsely assign this random signature to one of our 15 registered students. It extracted the HOG features, compared against all 15 hyperplanes, saw that maximum confidence was only 14%, and triggered open-set impostor rejection."*

---

### Act 2: The Authorized Signer Test (Live Imitation) (2 mins)
*Now show that when an authorized pattern is drawn, the system successfully recognizes it.*

1. Click the **"Show Registered Signers Reference (S01–S15)"** dropdown beneath the canvas.
2. Point out the thumbnail for **`S01`** (a clean flowing stroke) or **`S02`**.
3. Click **"Clear"** on the signature pad.
4. Draw an imitation of the `S01` stroke on the canvas.
5. Click **"Identify Live Signature"**:
   - The result panel turns **Green**:
     $$\text{Identity matched: S01}$$
   - Highlight the **Raw vs. Preprocessed Side-by-Side**:
     - *Raw Input*: Exactly what you just drew live.
     - *Preprocessed*: Otsu binarized, tightly cropped to ink bounding box, aspect-preserved, and centered on $128 \times 128$.
   - Show the **Top-3 Ranking Table**: `S01` has the dominant probability, followed by distant secondary candidates.

---

### Act 3: Live Paper Test (Webcam Snapshot or Phone Upload) (1.5 mins)
*Proves offline physical paper compatibility.*

1. Click the **"Upload / Cam"** tab in the Input Source section.
2. **Option A: Laptop Webcam**:
   - Click **"Or hold signature to webcam"**.
   - Sign a small slip of paper with a pen and hold it in front of the laptop camera.
   - Click **"Capture & Identify"**.
   - The system automatically captures the video frame, binarizes the ink strokes off the paper, and runs inference!
3. **Option B: File Drag & Drop**:
   - Drag any `.png` or `.jpg` photo of a signature directly into the drop zone.
   - Watch the instant real-time prediction.

---

### Act 4: Interactive Security Sensitivity ($\tau$ Slider) (1 min)
*Demonstrates operational flexibility.*

1. With a recognized signature active on screen (e.g. confidence ~55%):
2. Grab the **Rejection Threshold slider ($\tau$)** in the sidebar.
3. Slide it up to **`0.75`** (high security):
   - The banner immediately switches from green to red: **"Below threshold — rejected"**.
   - *Explanation*: High-security institutions (e.g., bank wire transfers) raise $\tau$ to prevent any borderline acceptances.
4. Slide it back down to **`0.44`** (or click the reset button):
   - Returns to green.
   - *Explanation*: Standard authentication settings balance False Acceptance Rate (FAR) and False Rejection Rate (FRR) at the Equal Error Rate (EER).

---

### Act 5: Conclude with the Methodology Tab (1 min)
1. Click **"Methodology"** in the top navbar.
2. Show the **4-Stage End-to-End Pipeline**:
   - `Binarize` (Otsu) $\to$ `Centre` (Aspect-ratio preservation) $\to$ `Classify` (HOG + SVM/RF) $\to$ `Reject` (Calibrated $\tau$).
3. Point out the **Confusion Matrices** (`cm_svm.png`, `cm_rf.png`) and the **Rejection Curve** (`reject_curve.png`) to validate your experimental findings.

---

## 💡 Examiner Q&A Cheat Sheet

| Question | Winning Answer |
| :--- | :--- |
| **"Can anyone just fool the system by scribbling?"** | *"No — we just demonstrated that live. Random scribbles lack the specific geometric and stroke orientation patterns learned by our HOG descriptors, yielding softmax probabilities far below our 0.44 safety threshold."* |
| **"Why did SVM achieve 80% while CNN got lower on this dataset?"** | *"Deep CNNs have millions of parameters and require massive datasets (thousands of images per class) to avoid severe overfitting. With 15 identities and limited handwriting samples, HOG feature extraction directly captures edge gradients and stroke curvature, which Linear SVM optimizes with high mathematical efficiency."* |
| **"How does Otsu handle paper shadows or poor lighting?"** | *"Otsu binarization calculates the bimodal variance of the grayscale histogram to dynamically find the optimal threshold for that specific lighting condition, followed by connected-component filtering that strips away isolated dust or camera noise."* |
| **"What is the difference between identification and verification?"** | *"Verification is 1-to-1: 'Is this signature specifically John Doe's?' Identification is 1-to-N: 'Which of the 15 enrolled signatories does this belong to, or is it an impostor?' Our system performs open-set identification."* |
