"""
Script de détection en temps réel du langage des signes ASL
Utilise MediaPipe pour la détection de main et CNN pour la classification
Auteur: Expert IA
"""

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from collections import deque
import time

# Configuration
MODEL_PATH = 'model/asl_model.h5'
CLASS_NAMES_PATH = 'model/class_names.txt'
IMG_SIZE = 64
CONFIDENCE_THRESHOLD = 0.6
SMOOTHING_FRAMES = 10  # Nombre de frames pour le lissage

class ASLDetector:
    """
    Classe pour la détection et reconnaissance du langage des signes ASL en temps réel
    """
    
    def __init__(self):
        """
        Initialise le détecteur avec MediaPipe et le modèle CNN
        """
        print("🚀 Initialisation du détecteur ASL...")
        
        # Charger le modèle
        try:
            self.model = tf.keras.models.load_model(MODEL_PATH)
            print("✅ Modèle chargé avec succès")
        except Exception as e:
            print(f"❌ Erreur lors du chargement du modèle: {e}")
            raise
        
        # Charger les noms de classes
        try:
            with open(CLASS_NAMES_PATH, 'r') as f:
                self.class_names = [line.strip() for line in f.readlines()]
            print(f"✅ {len(self.class_names)} classes chargées")
        except Exception as e:
            print(f"❌ Erreur lors du chargement des classes: {e}")
            raise
        
        # Initialiser MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # File pour le lissage des prédictions
        self.prediction_queue = deque(maxlen=SMOOTHING_FRAMES)
        
        # Variables de performance
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        
        print("✅ Détecteur initialisé avec succès")
    
    def extract_hand_roi(self, frame, hand_landmarks):
        """
        Extrait la région d'intérêt (ROI) contenant la main
        
        Args:
            frame: Image d'entrée
            hand_landmarks: Landmarks de la main détectés par MediaPipe
        
        Returns:
            roi: Image recadrée et redimensionnée de la main
        """
        h, w, _ = frame.shape
        
        # Obtenir les coordonnées min/max des landmarks
        x_coords = [lm.x for lm in hand_landmarks.landmark]
        y_coords = [lm.y for lm in hand_landmarks.landmark]
        
        # Calculer le bounding box avec une marge
        margin = 0.2
        x_min = max(0, int((min(x_coords) - margin) * w))
        x_max = min(w, int((max(x_coords) + margin) * w))
        y_min = max(0, int((min(y_coords) - margin) * h))
        y_max = min(h, int((max(y_coords) + margin) * h))
        
        # Extraire la ROI
        roi = frame[y_min:y_max, x_min:x_max]
        
        # Éviter les ROI vides
        if roi.size == 0:
            return None
        
        # Redimensionner et normaliser
        roi = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
        roi = roi / 255.0
        
        return roi, (x_min, y_min, x_max, y_max)
    
    def get_smoothed_prediction(self, predictions):
        """
        Applique un lissage sur les prédictions pour réduire le bruit
        
        Args:
            predictions: Prédictions du modèle pour la frame actuelle
        
        Returns:
            class_name: Classe prédite lissée
            confidence: Confiance de la prédiction
        """
        # Ajouter les prédictions à la queue
        predicted_class = np.argmax(predictions)
        self.prediction_queue.append(predicted_class)
        
        # Calculer la classe la plus fréquente
        if len(self.prediction_queue) > 0:
            most_common = max(set(self.prediction_queue), 
                            key=list(self.prediction_queue).count)
            confidence = predictions[0][most_common]
            return self.class_names[most_common], confidence
        
        return self.class_names[predicted_class], predictions[0][predicted_class]
    
    def draw_info(self, frame, class_name, confidence, bbox=None):
        """
        Affiche les informations sur la frame
        
        Args:
            frame: Image d'entrée
            class_name: Nom de la classe prédite
            confidence: Confiance de la prédiction
            bbox: Coordonnées du bounding box (x_min, y_min, x_max, y_max)
        """
        h, w, _ = frame.shape
        
        # Dessiner le bounding box si disponible
        if bbox:
            x_min, y_min, x_max, y_max = bbox
            color = (0, 255, 0) if confidence > CONFIDENCE_THRESHOLD else (0, 165, 255)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
        
        # Préparer le texte
        if confidence > CONFIDENCE_THRESHOLD and class_name != 'nothing':
            text = f"Lettre: {class_name.upper()}"
            conf_text = f"Confiance: {confidence:.2%}"
            color = (0, 255, 0)
        else:
            text = "Aucune lettre détectée"
            conf_text = ""
            color = (0, 165, 255)
        
        # Fond pour le texte principal
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)[0]
        cv2.rectangle(frame, (10, 10), (20 + text_size[0], 50 + text_size[1]), 
                     (0, 0, 0), -1)
        cv2.putText(frame, text, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                   1.2, color, 2)
        
        # Afficher la confiance
        if conf_text:
            cv2.putText(frame, conf_text, (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (255, 255, 255), 2)
        
        # Afficher le FPS
        fps_text = f"FPS: {self.fps:.1f}"
        cv2.putText(frame, fps_text, (w - 150, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (255, 255, 0), 2)
        
        # Instructions
        instructions = "Appuyez sur 'Q' pour quitter"
        cv2.putText(frame, instructions, (15, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, (200, 200, 200), 1)
    
    def update_fps(self):
        """
        Met à jour le calcul du FPS
        """
        self.frame_count += 1
        if self.frame_count % 30 == 0:
            elapsed = time.time() - self.start_time
            self.fps = 30 / elapsed
            self.start_time = time.time()
    
    def run(self):
        """
        Exécute la boucle principale de détection en temps réel
        """
        print("\n📹 Démarrage de la caméra...")
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Impossible d'ouvrir la caméra")
            return
        
        # Configuration de la caméra
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("✅ Caméra démarrée")
        print("\n🎯 Détection en cours...")
        print("   Montrez votre main à la caméra")
        print("   Appuyez sur 'Q' pour quitter\n")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Erreur lors de la lecture de la frame")
                break
            
            # Inverser horizontalement pour effet miroir
            frame = cv2.flip(frame, 1)
            
            # Convertir en RGB pour MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Détecter les mains
            results = self.hands.process(rgb_frame)
            
            class_name = "nothing"
            confidence = 0.0
            bbox = None
            
            # Si une main est détectée
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Dessiner les landmarks
                    self.mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style()
                    )
                    
                    # Extraire la ROI de la main
                    result = self.extract_hand_roi(frame, hand_landmarks)
                    
                    if result is not None:
                        roi, bbox = result
                        
                        # Prédiction
                        roi_batch = np.expand_dims(roi, axis=0)
                        predictions = self.model.predict(roi_batch, verbose=0)
                        
                        # Obtenir la prédiction lissée
                        class_name, confidence = self.get_smoothed_prediction(predictions)
            
            # Afficher les informations
            self.draw_info(frame, class_name, confidence, bbox)
            
            # Mettre à jour le FPS
            self.update_fps()
            
            # Afficher la frame
            cv2.imshow('ASL Sign Language Detection', frame)
            
            # Quitter avec 'Q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n👋 Arrêt de la détection...")
                break
        
        # Libérer les ressources
        cap.release()
        cv2.destroyAllWindows()
        self.hands.close()
        print("✅ Ressources libérées")
        print("🎉 Programme terminé")

def main():
    """
    Fonction principale
    """
    print("=" * 60)
    print("   🤟 Détecteur de Langage des Signes ASL en Temps Réel")
    print("=" * 60)
    
    try:
        detector = ASLDetector()
        detector.run()
    except KeyboardInterrupt:
        print("\n\n⚠️ Interruption par l'utilisateur")
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()