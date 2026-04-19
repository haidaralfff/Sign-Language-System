"""
Real-time gesture recognition inference using webcam.

Features:
- Live video capture from webcam
- Region of Interest (ROI) detection box
- Gesture classification with confidence scores
- Top-3 predictions display
- Keyboard controls: q=quit, s=save snapshot, r=reset
- Preprocessing: resize, normalize, CLAHE, Gaussian blur
"""

import logging
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Dict, Tuple, Optional
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GestureRecognizer:
    """Real-time gesture recognition using trained CNN model."""
    
    def __init__(
        self,
        model_path: str = 'model/cnn_model.h5',
        label_dict: Optional[Dict[str, int]] = None,
        img_size: int = 64,
        grayscale: bool = False,
        apply_clahe: bool = True,
        apply_blur: bool = True,
    ):
        """
        Initialize gesture recognizer.
        
        Args:
            model_path (str): Path to saved model (H5 format)
            label_dict (Dict, optional): Class name to index mapping
            img_size (int): Input image size (default 64)
            grayscale (bool): Use grayscale mode (default False for RGB)
            apply_clahe (bool): Apply CLAHE contrast enhancement
            apply_blur (bool): Apply Gaussian blur
        """
        self.model_path = Path(model_path)
        self.img_size = img_size
        self.grayscale = grayscale
        self.apply_clahe = apply_clahe
        self.apply_blur = apply_blur
        
        # Default 26-letter alphabet mapping
        self.label_dict = label_dict or {
            letter: idx for idx, letter in enumerate('ABCDEFGHIKLMNOPQRSTUVWXYZ')
        }
        self.idx_to_class = {v: k for k, v in self.label_dict.items()}
        
        # Load model
        self._load_model()
        
        # CLAHE object for contrast enhancement
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) \
            if apply_clahe else None
        
        logger.info(f"GestureRecognizer initialized:")
        logger.info(f"  - Model: {self.model_path}")
        logger.info(f"  - Image size: {img_size}x{img_size}")
        logger.info(f"  - Grayscale: {grayscale}")
        logger.info(f"  - Classes: {len(self.label_dict)}")
    
    def _load_model(self) -> None:
        """Load trained model from disk."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        self.model = tf.keras.models.load_model(str(self.model_path))
        logger.info(f"Model loaded from: {self.model_path}")
    
    def preprocess_image(self, img: np.ndarray) -> np.ndarray:
        """
        Preprocess image for model prediction.
        
        Args:
            img (np.ndarray): Input image (BGR format from OpenCV)
            
        Returns:
            np.ndarray: Preprocessed image, shape (1, img_size, img_size, channels)
        """
        # Resize
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        
        # Convert to grayscale if needed
        if self.grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Apply CLAHE
        if self.apply_clahe:
            if self.grayscale:
                img = self.clahe.apply(img)
            else:
                for i in range(3):
                    img[:, :, i] = self.clahe.apply(img[:, :, i])
        
        # Apply Gaussian blur
        if self.apply_blur:
            img = cv2.GaussianBlur(img, (3, 3), 0)
        
        # Normalize
        img = img.astype(np.float32) / 255.0
        
        # Add channel dimension if grayscale
        if self.grayscale:
            img = np.expand_dims(img, axis=-1)
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        return img
    
    def predict(self, img: np.ndarray, top_k: int = 3) -> Tuple[str, float, list]:
        """
        Predict gesture class from image.
        
        Args:
            img (np.ndarray): Input image (BGR from OpenCV)
            top_k (int): Return top-k predictions
            
        Returns:
            Tuple[str, float, list]:
                - predicted_class: Most likely class name
                - confidence: Confidence score (0-1)
                - top_k_predictions: List of (class, score) tuples
        """
        # Preprocess
        img_preprocessed = self.preprocess_image(img)
        
        # Assert shape
        assert img_preprocessed.shape[0] == 1, f"Batch size must be 1, got {img_preprocessed.shape[0]}"
        assert img_preprocessed.shape[1] == self.img_size, f"Height mismatch"
        assert img_preprocessed.shape[2] == self.img_size, f"Width mismatch"
        
        # Predict
        predictions = self.model.predict(img_preprocessed, verbose=0)[0]
        
        # Get top-k predictions
        top_indices = np.argsort(predictions)[-top_k:][::-1]
        top_k_preds = [
            (self.idx_to_class[idx], float(predictions[idx]))
            for idx in top_indices
        ]
        
        # Get top-1 prediction
        predicted_idx = np.argmax(predictions)
        predicted_class = self.idx_to_class[predicted_idx]
        confidence = float(predictions[predicted_idx])
        
        return predicted_class, confidence, top_k_preds
    
    def draw_roi_box(
        self,
        frame: np.ndarray,
        roi_top: int = 100,
        roi_left: int = 200,
        roi_height: int = 200,
        roi_width: int = 200,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        Draw ROI box on frame.
        
        Args:
            frame (np.ndarray): Input frame
            roi_top (int): ROI top-left y coordinate
            roi_left (int): ROI top-left x coordinate
            roi_height (int): ROI height
            roi_width (int): ROI width
            color (Tuple): Box color (BGR)
            thickness (int): Box line thickness
            
        Returns:
            Tuple[np.ndarray, Tuple]: (frame_with_box, roi_coords)
        """
        roi_bottom = roi_top + roi_height
        roi_right = roi_left + roi_width
        
        # Draw rectangle
        cv2.rectangle(frame, (roi_left, roi_top), (roi_right, roi_bottom), color, thickness)
        
        # Draw corner brackets
        corner_len = 20
        corners = [
            ((roi_left, roi_top), (roi_left + corner_len, roi_top), (roi_left, roi_top + corner_len)),
            ((roi_right, roi_top), (roi_right - corner_len, roi_top), (roi_right, roi_top + corner_len)),
            ((roi_left, roi_bottom), (roi_left + corner_len, roi_bottom), (roi_left, roi_bottom - corner_len)),
            ((roi_right, roi_bottom), (roi_right - corner_len, roi_bottom), (roi_right, roi_bottom - corner_len)),
        ]
        
        for pts in corners:
            cv2.line(frame, pts[0], pts[1], color, thickness)
            cv2.line(frame, pts[0], pts[2], color, thickness)
        
        return frame, (roi_left, roi_top, roi_right, roi_bottom)
    
    def draw_predictions(
        self,
        frame: np.ndarray,
        predicted_class: str,
        confidence: float,
        top_k_preds: list,
        position: Tuple[int, int] = (10, 40),
        font_scale: float = 1.5
    ) -> np.ndarray:
        """
        Draw predictions on frame.
        
        Args:
            frame (np.ndarray): Input frame
            predicted_class (str): Predicted class name
            confidence (float): Confidence score
            top_k_preds (list): Top-k predictions
            position (Tuple): Text position (x, y)
            font_scale (float): Font size scale
            
        Returns:
            np.ndarray: Frame with predictions drawn
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 2
        
        # Draw main prediction
        text = f"{predicted_class} ({confidence*100:.1f}%)"
        cv2.putText(
            frame, text,
            position,
            font, font_scale,
            (0, 255, 0),  # Green
            thickness
        )
        
        # Draw top-3 predictions
        for i, (class_name, score) in enumerate(top_k_preds[:3]):
            y_offset = position[1] + (i + 1) * 35
            bar_width = int(score * 150)
            
            # Prediction text
            pred_text = f"{class_name}: {score*100:.1f}%"
            cv2.putText(
                frame, pred_text,
                (position[0] + 10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 0),  # Cyan
                1
            )
            
            # Confidence bar background
            cv2.rectangle(
                frame,
                (position[0] + 200, y_offset - 15),
                (position[0] + 350, y_offset),
                (50, 50, 50),  # Dark gray
                -1
            )
            
            # Confidence bar fill
            color = (0, 255, 0) if score > 0.7 else (0, 165, 255) if score > 0.5 else (0, 0, 255)
            cv2.rectangle(
                frame,
                (position[0] + 200, y_offset - 15),
                (position[0] + 200 + bar_width, y_offset),
                color,
                -1
            )
        
        return frame
    
    def run_webcam(
        self,
        camera_id: int = 0,
        roi_height: int = 200,
        roi_width: int = 200,
        fps_limit: int = 30,
        snapshot_dir: str = 'snapshots'
    ) -> None:
        """
        Run real-time gesture recognition from webcam.
        
        Controls:
        - 'q': Quit
        - 's': Save snapshot
        - 'r': Reset/Show instructions
        
        Args:
            camera_id (int): Camera device ID
            roi_height (int): ROI box height
            roi_width (int): ROI box width
            fps_limit (int): Frame rate limit
            snapshot_dir (str): Directory to save snapshots
        """
        Path(snapshot_dir).mkdir(parents=True, exist_ok=True)
        
        # Open camera
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            logger.error(f"Failed to open camera {camera_id}")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, fps_limit)
        
        logger.info("Starting real-time gesture recognition...")
        logger.info("Controls: 'q'=quit, 's'=save snapshot, 'r'=reset")
        
        show_instructions = True
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to read frame")
                break
            
            frame_count += 1
            
            # Flip frame horizontally for selfie view
            frame = cv2.flip(frame, 1)
            
            h, w = frame.shape[:2]
            roi_top = (h - roi_height) // 2
            roi_left = (w - roi_width) // 2
            
            # Draw ROI box
            frame, roi_coords = self.draw_roi_box(
                frame,
                roi_top=roi_top,
                roi_left=roi_left,
                roi_height=roi_height,
                roi_width=roi_width
            )
            
            # Extract ROI
            roi = frame[
                roi_coords[1]:roi_coords[3],
                roi_coords[0]:roi_coords[2]
            ]
            
            # Predict
            try:
                predicted_class, confidence, top_k_preds = self.predict(roi)
                
                # Draw predictions
                frame = self.draw_predictions(
                    frame,
                    predicted_class,
                    confidence,
                    top_k_preds,
                    position=(10, 40)
                )
            except Exception as e:
                logger.warning(f"Prediction error: {e}")
            
            # Draw FPS
            cv2.putText(
                frame,
                f"Frame: {frame_count}",
                (w - 150, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                1
            )
            
            # Draw instructions
            if show_instructions:
                instructions = [
                    "CONTROLS:",
                    "q = Quit",
                    "s = Save snapshot",
                    "r = Reset info"
                ]
                for i, text in enumerate(instructions):
                    cv2.putText(
                        frame,
                        text,
                        (10, h - 80 + i*25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (200, 200, 200),
                        1
                    )
            
            # Display frame
            cv2.imshow('SIBI Gesture Recognition - Real-time Inference', frame)
            
            # Keyboard controls
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):  # Quit
                logger.info("Exiting...")
                break
            elif key == ord('s'):  # Save snapshot
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                snapshot_path = Path(snapshot_dir) / f"gesture_{predicted_class}_{timestamp}.png"
                cv2.imwrite(str(snapshot_path), frame)
                logger.info(f"Snapshot saved: {snapshot_path}")
            elif key == ord('r'):  # Reset/Show instructions
                show_instructions = not show_instructions
        
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Real-time inference stopped.")


def main():
    """Main entry point for real-time inference."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Real-time SIBI gesture recognition'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='model/cnn_model.h5',
        help='Path to trained model'
    )
    parser.add_argument(
        '--camera',
        type=int,
        default=0,
        help='Camera device ID'
    )
    parser.add_argument(
        '--img-size',
        type=int,
        default=64,
        help='Input image size'
    )
    parser.add_argument(
        '--grayscale',
        action='store_true',
        help='Use grayscale mode'
    )
    parser.add_argument(
        '--no-clahe',
        action='store_true',
        help='Disable CLAHE enhancement'
    )
    parser.add_argument(
        '--no-blur',
        action='store_true',
        help='Disable Gaussian blur'
    )
    parser.add_argument(
        '--roi-size',
        type=int,
        default=200,
        help='ROI box size'
    )
    
    args = parser.parse_args()
    
    # Initialize recognizer
    recognizer = GestureRecognizer(
        model_path=args.model,
        img_size=args.img_size,
        grayscale=args.grayscale,
        apply_clahe=not args.no_clahe,
        apply_blur=not args.no_blur,
    )
    
    # Run webcam inference
    recognizer.run_webcam(
        camera_id=args.camera,
        roi_height=args.roi_size,
        roi_width=args.roi_size
    )


if __name__ == "__main__":
    main()
