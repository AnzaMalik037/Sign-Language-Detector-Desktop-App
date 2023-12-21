import pickle
import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
import pygame
import os

class SignLanguageClassifier:
    def __init__(self, model_path='./model.p', capture_device=0):
        self.model_dict = pickle.load(open(model_path, 'rb'))
        self.model = self.model_dict['model']
        self.cap = cv2.VideoCapture(capture_device)
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
        self.labels_dict = {
            0: {'label': 'Bad', 'audio': 'audio/bad.mp3'},
            1: {'label': 'Rock/Yes', 'audio': 'audio/rock2.mp3'},
            2: {'label': 'Paper/Hello', 'audio': 'audio/paper2.mp3'},
            3: {'label': 'Scissor', 'audio': 'audio/scissor.mp3'},
            4: {'label': 'Good/Okey', 'audio': 'audio/good.mp3'},
            5: {'label': 'ILoveYou', 'audio': 'audio/ILY.mp3'},
            6: {'label': 'No', 'audio': 'audio/no.mp3'},
            7: {'label': 'Help', 'audio': 'audio/help.mp3'},
            8: {'label': 'Im angry', 'audio': 'audio/angry.mp3'},
            9: {'label': 'Im happy', 'audio': 'audio/happy.mp3'},
            10: {'label': 'So Delicious', 'audio': 'audio/delicious.mp3'}
        }

        # Create Tkinter GUI
        self.root = tk.Tk()
        self.root.title("Sign Language Detector")
        self.root.geometry("400x300")

        # Create a canvas covering the entire window with a colored background
        self.canvas = tk.Canvas(self.root, width=400, height=300, bg="#ffef96")
        self.canvas.pack()

        # Add a title to the canvas with bold font
        title_text = "Sign Language Detector"
        title_font = ("Arial", 20, "bold")
        self.canvas.create_text(200, 50, text=title_text, font=title_font, fill="#50394c")

        # Add a "Start Detecting" button
        self.start_button = tk.Button(self.root, text="Start Detecting", command=self.run_classification, bg="#50394c", fg="#ffef96", font=("Arial", 12))
        self.start_button.place(relx=0.5, rely=0.4, anchor="center")

        # Add an "End Detection" button
        self.end_button = tk.Button(self.root, text="End Detection", command=self.end_detection, bg="#ff4949", fg="#ffffff", font=("Arial", 12))
        self.end_button.place(relx=0.5, rely=0.6, anchor="center")

        # Initialize Pygame for audio playback
        pygame.mixer.init()
        self.sounds = {index: pygame.mixer.Sound(audio_path) for index, data in self.labels_dict.items() for audio_path in [os.path.join(os.path.dirname(__file__), data['audio'])]}
        
        #error handeling basic
        if not self.cap.isOpened():
            print("Error: Unable to open video capture.")

    def run_classification(self):
        while True:
            data_aux = []
            x_ = []
            y_ = []

            ret, frame = self.cap.read()
            H, W, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = self.hands.process(frame_rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style())

                for hand_landmarks in results.multi_hand_landmarks:
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y

                        x_.append(x)
                        y_.append(y)

                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        data_aux.append(x - min(x_))
                        data_aux.append(y - min(y_))

                # Ensure that data_aux has a length of 84
                data_aux = data_aux[:84] + [0] * max(0, 84 - len(data_aux))

                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10

                x2 = int(max(x_) * W) - 10
                y2 = int(max(y_) * H) - 10

                prediction = self.model.predict([np.asarray(data_aux)])

                predicted_character = int(prediction[0])
                self.play_sound(predicted_character)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(frame, self.labels_dict[predicted_character]['label'], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def play_sound(self, character):
        # Stop any currently playing sound
        pygame.mixer.stop()

        if character in self.sounds:
            self.sounds[character].play()

    def end_detection(self):
        self.root.destroy()

if __name__ == "__main__":
    classifier = SignLanguageClassifier()
    classifier.root.mainloop()
