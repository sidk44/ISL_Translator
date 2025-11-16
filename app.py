import os
import cv2
import numpy as np
import pandas as pd
import csv
import argparse
import logging
import pyttsx3
import mediapipe as mp
from tensorflow.keras.models import load_model, Sequential, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model', 'keypoint_classifier')
CSV_PATH = os.path.join(MODEL_DIR, 'keypoint.csv')
MODEL_PATH = os.path.join(MODEL_DIR, 'keypoint_classifier.hdf5')
LE_PATH = os.path.join(MODEL_DIR, 'label_encoder.npy')

# Ensure directories exist
os.makedirs(MODEL_DIR, exist_ok=True)

# Mediapipe setup (hidden from UI)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Translations dictionary
translations = {
    "Hello": {"en": "Hello", "hi": "नमस्ते", "kn": "ಹಲೋ"},
    
}

# Utility: extract 42 keypoints

def calculate_keypoints(landmarks, shape):
    h, w = shape[:2]
    pts = []
    for lm in landmarks.landmark:
        x = min(int(lm.x * w), w - 1)
        y = min(int(lm.y * h), h - 1)
        pts.extend([x, y])
    return np.array(pts)

# Modern clean UI drawing

def draw_ui(frame, pts, label, probs=None, lang_txt=None, fps=None):
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # 1) Minimal dashed bounding box
    coords = pts.reshape(-1, 2)
    min_x, max_x = coords[:,0].min(), coords[:,0].max()
    min_y, max_y = coords[:,1].min(), coords[:,1].max()
    dash = 10; gap = 8; col_box = (0, 0, 255)
    # horizontal
    for x in range(int(min_x), int(max_x), dash+gap):
        cv2.line(overlay, (x, int(min_y)), (x+dash, int(min_y)), col_box, 2)
        cv2.line(overlay, (x, int(max_y)), (x+dash, int(max_y)), col_box, 2)
    # vertical
    for y in range(int(min_y), int(max_y), dash+gap):
        cv2.line(overlay, (int(min_x), y), (int(min_x), y+dash), col_box, 2)
        cv2.line(overlay, (int(max_x), y), (int(max_x), y+dash), col_box, 2)

    # 2) Smooth circular markers at landmarks
    for (x, y) in coords:
        cv2.circle(overlay, (x, y), 5, (255, 100, 100), -1)
        cv2.circle(overlay, (x, y), 8, (255, 100, 100), 1)

    # Blend overlay
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # 3) Clean translucent sidebar
    sidebar_w = 200
    cv2.rectangle(frame, (w-sidebar_w, 0), (w, h), (20, 20, 20), -1)

    # 4) Text elements
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Display gesture in red
    cv2.putText(frame, f'Gesture: {label}', (20, 40), font, 1.2, (0, 0, 255), 2, cv2.LINE_AA)
    if lang_txt:
        cv2.putText(frame, lang_txt, (20, 80), font, 0.9, (200,200,255), 2, cv2.LINE_AA)

    # 5) Top-3 probabilities bar chart
    if probs:
        for i, (cls, p) in enumerate(probs):
            y = 120 + i*30
            text = f'{cls}: {int(p*100)}%'
            cv2.putText(frame, text, (w-sidebar_w+10, y), font, 0.7, (220,220,220), 1)
            cv2.rectangle(frame, (w-sidebar_w+110, y-15), (w-sidebar_w+110+int((sidebar_w-130)*p), y+5), (100,255,100), -1)

    # 6) FPS display
    if fps:
        cv2.putText(frame, f'FPS: {fps:.1f}', (20, h-20), font, 0.7, (240,240,200), 2, cv2.LINE_AA)

# Record mode

def record(label):
    cap = cv2.VideoCapture(0)
    print(f"Recording '{label}'. Press 'q' to stop.")
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)
        if res.multi_hand_landmarks:
            pts = calculate_keypoints(res.multi_hand_landmarks[0], frame.shape)
            with open(CSV_PATH, 'a', newline='') as f:
                csv.writer(f).writerow([label] + pts.tolist())
            draw_ui(frame, pts, label)
        cv2.imshow('Record', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cap.release(); cv2.destroyAllWindows()

# Build MLP

def build_model(input_dim, num_classes):
    net = Sequential([Dense(128, activation='relu', input_shape=(input_dim,)),
                      Dense(64, activation='relu'),
                      Dense(num_classes, activation='softmax')])
    net.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return net

# Train mode

def train_model(full=False):
    df = pd.read_csv(CSV_PATH)
    X, y = df.iloc[:,1:].values, df.iloc[:,0].values
    le = LabelEncoder(); y_enc = le.fit_transform(y)
    classes_new = le.classes_; np.save(LE_PATH, classes_new)
    y_cat = to_categorical(y_enc, len(classes_new))
    X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)
    if full or not os.path.exists(MODEL_PATH):
        net = build_model(X_train.shape[1], len(classes_new))
    else:
        old_classes = np.load(LE_PATH, allow_pickle=True)
        net = load_model(MODEL_PATH)
        if len(classes_new) != len(old_classes):
            x = net.layers[-2].output
            net = Model(inputs=net.input, outputs=Dense(len(classes_new), activation='softmax')(x))
            net.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    net.fit(X_train, y_train, epochs=20 if not full else 50, batch_size=32, validation_data=(X_test, y_test))
    net.save(MODEL_PATH); logging.info('Model saved.')
LETTERS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
import time

def recognize(lang='en', voice=False):
    net = load_model(MODEL_PATH)
    classes = np.load(LE_PATH, allow_pickle=True)
    le = LabelEncoder(); le.classes_ = classes
    engine = pyttsx3.init() if voice else None
    langs = ['en','hi','kn']; idx = langs.index(lang)
    cap = cv2.VideoCapture(0)
    prev = cv2.getTickCount()





    # --- For word accumulation & debouncing ---
    current_word = []
    last_letter = None
    last_detected_time = time.time()
    CLEAR_DELAY = 2  # seconds to clear word



    # --- For stable (debounced) letter display ---
    stable_letter = None
    stable_count = 0
    STABLE_FRAMES = 10  # adjust for how steady the hand sign should be

    font = cv2.FONT_HERSHEY_SIMPLEX

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame,1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)
        fps = cv2.getTickFrequency()/(cv2.getTickCount()-prev); prev=cv2.getTickCount()

        detected_letter = None
        if res.multi_hand_landmarks:
            for hand_landmarks in res.multi_hand_landmarks:
                pts = calculate_keypoints(hand_landmarks, frame.shape)
                preds = net.predict(pts.reshape(1,-1))[0]
                top = sorted(enumerate(preds), key=lambda x: x[1], reverse=True)[:3]
                top_probs = [(classes[i], p) for i,p in top]
                lbl = top_probs[0][0]

                # Buffer only letter gestures from the first hand
                if lbl in LETTERS and detected_letter is None:
                    detected_letter = lbl

                lang_txt = translations.get(lbl,{}).get(langs[idx], '')
                if engine and lang_txt: engine.say(lang_txt); engine.runAndWait()
                draw_ui(frame, pts, lbl, top_probs, lang_txt, fps)

        else:
            lbl = ""

        # --- Debouncing logic for stable letter display ---
        if detected_letter:
            if detected_letter == stable_letter:
                stable_count += 1
            else:
                stable_letter = detected_letter
                stable_count = 1
            last_detected_time = time.time()
        else:
            stable_letter = None
            stable_count = 0

        # Only allow adding a letter to word if stable for enough frames and is new
        display_letter = stable_letter if stable_count >= STABLE_FRAMES else ""

        if display_letter and (not current_word or current_word[-1] != display_letter):
            current_word.append(display_letter)
            last_letter = display_letter
            last_detected_time = time.time()

        now = time.time()
        # Clear word after inactivity
        if now - last_detected_time > CLEAR_DELAY:
            current_word = []
            last_letter = None

        # --- Display the current word at the bottom (left) ---
        word_str = ''.join(current_word)
        if word_str:
            h, w = frame.shape[:2]
            cv2.putText(frame, word_str, (50, h-40), font, 2, (0,255,200), 4, cv2.LINE_AA)

        # --- Show keyboard shortcuts at bottom right ---
        h, w = frame.shape[:2]
        kb_shortcuts = "q:QUIT  l:LANG  s:SPACE  c:CLEAR"
        (text_width, text_height), _ = cv2.getTextSize(kb_shortcuts, font, 0.7, 2)
        cv2.putText(frame, kb_shortcuts, (w-text_width-20, h-20), font, 0.7, (200,220,220), 2, cv2.LINE_AA)

        cv2.imshow('Recognition', frame)
        k = cv2.waitKey(1)&0xFF
        if k==ord('q'): 
            break
        elif k==ord('l'): 
            idx=(idx+1)%len(langs)
        elif k==ord('s'):  # Add space
            if not current_word or current_word[-1] != ' ':
                current_word.append(' ')
            last_letter = None
            last_detected_time = time.time()
        elif k==ord('c'):  # Clear word
            current_word = []
            last_letter = None

    cap.release(); cv2.destroyAllWindows()



# CLI entrypoint

def main():
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest='cmd')
    sp.add_parser('record').add_argument('label')
    t = sp.add_parser('train'); t.add_argument('--full', action='store_true')
    r = sp.add_parser('recognize'); r.add_argument('--lang', choices=['en','hi','kn'], default='en'); r.add_argument('--voice', action='store_true')
    args = p.parse_args()
    if args.cmd=='record': record(args.label)
    elif args.cmd=='train': train_model(full=args.full)
    elif args.cmd=='recognize': recognize(args.lang, args.voice)
    else: p.print_help()

if __name__=='__main__': main()
