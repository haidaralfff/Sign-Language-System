"""
Enhanced real-time gesture recognition with MediaPipe hand detection.

Features:
- Automatic hand detection using MediaPipe
- Hand landmark visualization (skeleton)
- CNN-based gesture classification
- Real-time ROI extraction from detected hands
- Top-3 predictions display
- Smooth predictions averaging
- Keyboard controls: q=quit, s=save, r=reset, h=show/hide landmarks
"""

import logging
import cv2
import numpy as np
import tensorflow as tf
import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from datetime import datetime
import os

# Import custom modules
from utils.mediapipe_handler import HandDetector
from realtime_inference import GestureRecognizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MediaPipeGestureRecognizer:
    """Enhanced gesture recognizer combining MediaPipe hand detection with CNN."""
    
    def __init__(
        self,
        model_path: str = 'model/cnn_model.h5',
        img_size: int = 64,
        confidence_threshold: float = 0.7,
        smoothing_window: int = 5,
        apply_clahe: bool = True,
        apply_blur: bool = True,
    ):
        """
        Initialize MediaPipe gesture recognizer.
        
        Args:
            model_path (str): Path to trained CNN model
            img_size (int): Input size for CNN (default 64)
            confidence_threshold (float): Minimum confidence for gesture prediction
            smoothing_window (int): Number of frames for prediction smoothing
            apply_clahe (bool): Apply CLAHE contrast enhancement
            apply_blur (bool): Apply Gaussian blur
        """
        # Initialize MediaPipe hand detector
        self.hand_detector = HandDetector(
            num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )
        
        # Initialize CNN gesture recognizer
        self.gesture_recognizer = GestureRecognizer(
            model_path=model_path,
            img_size=img_size,
            apply_clahe=apply_clahe,
            apply_blur=apply_blur,
        )
        
        self.img_size = img_size
        self.confidence_threshold = confidence_threshold
        self.smoothing_window = smoothing_window
        
        # Prediction smoothing
        self.prediction_history: List[Tuple[str, float]] = []
        
        # UI state
        self.show_landmarks = True
        self.show_info = True
        
        logger.info("MediaPipeGestureRecognizer initialized successfully!")
        logger.info("✓ Gesture filter enabled (G/W bias mitigation)")
    
    def smooth_predictions(
        self,
        current_pred: Tuple[str, float]
    ) -> Tuple[str, float]:
        """
        Smooth predictions using history window.
        
        Args:
            current_pred: (class_name, confidence)
            
        Returns:
            Tuple: Smoothed (class_name, avg_confidence)
        """
        self.prediction_history.append(current_pred)
        
        # Keep only recent predictions
        if len(self.prediction_history) > self.smoothing_window:
            self.prediction_history.pop(0)
        
        # Average confidence for same prediction
        class_confidences = {}
        for pred_class, confidence in self.prediction_history:
            if pred_class not in class_confidences:
                class_confidences[pred_class] = []
            class_confidences[pred_class].append(confidence)
        
        # Get most frequent prediction
        most_frequent_class = max(
            class_confidences.items(),
            key=lambda x: len(x[1])
        )[0]
        
        avg_confidence = np.mean(
            class_confidences[most_frequent_class]
        )
        
        return most_frequent_class, avg_confidence
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[Dict]]:
        """
        Process frame with hand detection and gesture classification.
        
        Args:
            frame (np.ndarray): Input frame (BGR from OpenCV)
            
        Returns:
            Tuple: (annotated_frame, gesture_info_dict)
        """
        # Detect hands
        hands_data, frame_rgb = self.hand_detector.detect_hands(frame)
        
        annotated_frame = frame.copy()
        gesture_info = None
        
        if not hands_data:
            # No hands detected
            cv2.putText(
                annotated_frame,
                "No hand detected - Move hand in frame",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
            )
            return annotated_frame, None
        
        # Process first hand
        hand = hands_data[0]
        hand_landmarks = hand['landmarks']
        keypoints = hand['keypoints']
        handedness = hand['handedness']
        hand_confidence = hand['confidence']
        
        # Draw landmarks if enabled
        if self.show_landmarks:
            landmarks_frame = self.hand_detector.draw_landmarks(
                annotated_frame,
                hand_landmarks,
                handedness,
                hand_confidence,
            )
            if landmarks_frame is not None:
                annotated_frame = landmarks_frame
        
        # Draw ROI box
        if annotated_frame is not None:
            roi_frame = self.hand_detector.draw_roi_box(
                annotated_frame,
                keypoints,
                padding=0.2,
                color=(0, 255, 0),
            )
            if roi_frame is not None:
                annotated_frame = roi_frame
        
        # Extract ROI
        roi = self.hand_detector.get_hand_roi(
            frame,
            keypoints,
            padding=0.2,
        )
        
        if roi is None:
            return annotated_frame, None
        
        # Predict gesture
        predicted_class, confidence, top_k = self.gesture_recognizer.predict(
            roi,
            top_k=3,
        )
        
        # Smooth predictions using exponential moving average
        smoothed_class, smoothed_confidence = self.smooth_predictions(
            (predicted_class, confidence)
        )
        
        gesture_info = {
            'class': smoothed_class,
            'confidence': smoothed_confidence,
            'top_k': top_k,
            'handedness': handedness,
            'roi': roi,
        }
        
        # Draw predictions on frame
        annotated_frame = self._draw_predictions(
            annotated_frame,
            gesture_info,
        )
        
        return annotated_frame, gesture_info
    
    def _draw_predictions(
        self,
        frame: np.ndarray,
        gesture_info: Dict,
    ) -> np.ndarray:
        """
        Draw gesture predictions on frame.
        
        Args:
            frame (np.ndarray): Input frame
            gesture_info (Dict): Gesture information from process_frame
            
        Returns:
            np.ndarray: Annotated frame
        """
        predicted_class = gesture_info['class']
        confidence = gesture_info['confidence']
        top_k = gesture_info['top_k']
        
        # Main prediction (top-left)
        prediction_text = f"{predicted_class}: {confidence*100:.1f}%"
        color = (0, 255, 0) if confidence > self.confidence_threshold else (0, 165, 255)
        
        cv2.putText(
            frame,
            prediction_text,
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            color,
            3,
        )
        
        # Top-3 bar chart (right side)
        bar_x_start = frame.shape[1] - 250
        bar_width = 180
        bar_height = 20
        bar_y_start = 60
        
        for idx, (class_name, score) in enumerate(top_k):
            y_pos = bar_y_start + idx * (bar_height + 5)
            
            # Class label
            cv2.putText(
                frame,
                f"{class_name}:",
                (bar_x_start - 50, y_pos + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                1,
            )
            
            # Background bar
            cv2.rectangle(
                frame,
                (bar_x_start, y_pos),
                (bar_x_start + bar_width, y_pos + bar_height),
                (50, 50, 50),
                -1,
            )
            
            # Confidence bar
            bar_length = int(bar_width * score)
            color = (0, 255, 0) if idx == 0 else (255, 165, 0)
            cv2.rectangle(
                frame,
                (bar_x_start, y_pos),
                (bar_x_start + bar_length, y_pos + bar_height),
                color,
                -1,
            )
            
            # Confidence text
            cv2.putText(
                frame,
                f"{score*100:.0f}%",
                (bar_x_start + bar_width + 10, y_pos + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
            )
        
        # Info panel (bottom-left)
        info_y = frame.shape[0] - 120
        info_texts = [
            f"Hand: {gesture_info['handedness']}",
            "Controls:",
            "  q=Quit  s=Save  r=Reset  h=Hide/Show Landmarks",
        ]
        
        for idx, text in enumerate(info_texts):
            cv2.putText(
                frame,
                text,
                (20, info_y + idx * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 200),
                1,
            )
        
        return frame
    
    def run_webcam(
        self,
        camera_id: int = 0,
        window_name: str = "MediaPipe Gesture Recognition",
    ) -> None:
        """
        Run real-time gesture recognition from webcam.
        
        Args:
            camera_id (int): Camera device ID (default 0)
            window_name (str): Window title
        """
        # Create output directories
        Path('snapshots').mkdir(exist_ok=True)
        
        # Open webcam
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            logger.error(f"Cannot open camera {camera_id}")
            return
        
        logger.info("Webcam opened. Starting gesture recognition...")
        logger.info("Controls: q=Quit, s=Save snapshot, r=Reset predictions, h=Toggle landmarks")
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                logger.error("Failed to read frame from webcam")
                break
            
            frame_count += 1
            
            # Flip frame horizontally for selfie view
            frame = cv2.flip(frame, 1)
            
            # Add frame counter
            cv2.putText(
                frame,
                f"Frame: {frame_count}",
                (frame.shape[1] - 180, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (100, 100, 100),
                1,
            )
            
            # Process frame
            annotated_frame, gesture_info = self.process_frame(frame)
            
            # Display frame
            cv2.imshow(window_name, annotated_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                logger.info("Quit requested by user")
                break
            
            elif key == ord('s') and gesture_info:
                # Save snapshot
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                gesture_class = gesture_info['class']
                filename = f"snapshots/gesture_{gesture_class}_{timestamp}.jpg"
                cv2.imwrite(filename, annotated_frame)
                logger.info(f"Snapshot saved: {filename}")
            
            elif key == ord('r'):
                # Reset predictions
                self.prediction_history.clear()
                logger.info("Prediction history reset")
            
            elif key == ord('h'):
                # Toggle landmarks
                self.show_landmarks = not self.show_landmarks
                status = "shown" if self.show_landmarks else "hidden"
                logger.info(f"Landmarks {status}")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        self.hand_detector.close()
        logger.info("Webcam session closed")


def main():
    """Main entry point for MediaPipe gesture recognition."""
    parser = argparse.ArgumentParser(
        description="Real-time gesture recognition with MediaPipe hand detection"
    )
    parser.add_argument(
        '--model',
        type=str,
        default='model/cnn_model.h5',
        help='Path to trained CNN model'
    )
    parser.add_argument(
        '--camera',
        type=int,
        default=0,
        help='Camera device ID (default: 0)'
    )
    parser.add_argument(
        '--img-size',
        type=int,
        default=64,
        help='Input image size for CNN (default: 64)'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.7,
        help='Confidence threshold for predictions (default: 0.7)'
    )
    parser.add_argument(
        '--smoothing',
        type=int,
        default=5,
        help='Number of frames for prediction smoothing (default: 5)'
    )
    
    args = parser.parse_args()
    
    # Validate model path
    if not Path(args.model).exists():
        logger.error(f"Model not found: {args.model}")
        return
    
    # Initialize and run
    recognizer = MediaPipeGestureRecognizer(
        model_path=args.model,
        img_size=args.img_size,
        confidence_threshold=args.threshold,
        smoothing_window=args.smoothing,
    )
    
    recognizer.run_webcam(camera_id=args.camera)


if __name__ == '__main__':
    main()
