"""
MediaPipe hand detection handler for gesture recognition.

Features:
- Real-time hand landmark detection
- Multiple hand detection (left/right)
- Hand confidence scoring
- Hand ROI extraction with padding
- Landmark visualization
- Smooth landmark filtering (optional)
"""

import logging
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HandDetector:
    """Real-time hand detection using MediaPipe with multiple hand support."""
    
    def __init__(
        self,
        num_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        static_image_mode: bool = False,
    ):
        """
        Initialize MediaPipe hand detector.
        
        Args:
            num_hands (int): Maximum number of hands to detect (1 or 2)
            min_detection_confidence (float): Minimum confidence threshold for hand detection
            min_tracking_confidence (float): Minimum confidence threshold for hand tracking
            static_image_mode (bool): If True, detects hands on every frame (slower but more accurate)
        """
        self.num_hands = max(1, min(num_hands, 2))
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.static_image_mode = static_image_mode
        
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=self.num_hands,
            model_complexity=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Hand landmarks history for smoothing (optional)
        self.landmarks_history = {}
        self.smoothing_window = 5
        
        logger.info(f"HandDetector initialized:")
        logger.info(f"  - Max hands: {self.num_hands}")
        logger.info(f"  - Detection confidence: {min_detection_confidence}")
        logger.info(f"  - Tracking confidence: {min_tracking_confidence}")
    
    def detect_hands(self, frame: np.ndarray) -> Tuple[List[Dict], np.ndarray]:
        """
        Detect hands and landmarks in frame.
        
        Args:
            frame (np.ndarray): Input frame (BGR from OpenCV)
            
        Returns:
            Tuple[List[Dict], np.ndarray]:
                - hands_data: List of detected hands with landmarks
                - frame_rgb: Frame converted to RGB (for MediaPipe)
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame with MediaPipe
        results = self.hands.process(frame_rgb)
        
        hands_data = []
        
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks, results.multi_handedness
            ):
                hand_info = {
                    'landmarks': hand_landmarks,
                    'handedness': handedness.classification[0].label,  # 'Left' or 'Right'
                    'confidence': handedness.classification[0].score,
                    'keypoints': self._extract_keypoints(hand_landmarks, frame.shape),
                }
                hands_data.append(hand_info)
        
        return hands_data, frame_rgb
    
    def _extract_keypoints(
        self,
        hand_landmarks: object,
        frame_shape: Tuple[int, int, int]
    ) -> np.ndarray:
        """
        Extract and normalize keypoints from hand landmarks.
        
        Args:
            hand_landmarks: MediaPipe hand landmarks object
            frame_shape: Shape of frame (height, width, channels)
            
        Returns:
            np.ndarray: Normalized keypoints, shape (21, 2)
        """
        height, width, _ = frame_shape
        keypoints = []
        
        for landmark in hand_landmarks.landmark:
            x = landmark.x * width
            y = landmark.y * height
            keypoints.append([x, y])
        
        return np.array(keypoints, dtype=np.float32)
    
    def get_hand_roi(
        self,
        frame: np.ndarray,
        keypoints: np.ndarray,
        padding: float = 0.2,
    ) -> Optional[np.ndarray]:
        """
        Extract hand Region of Interest (ROI) from frame.
        
        Args:
            frame (np.ndarray): Input frame
            keypoints (np.ndarray): Hand keypoints, shape (21, 2)
            padding (float): Padding ratio (0.2 = 20% padding)
            
        Returns:
            Optional[np.ndarray]: Cropped hand ROI or None if invalid
        """
        if keypoints is None or len(keypoints) < 21:
            return None
        
        # Get bounding box from keypoints
        x_min, y_min = keypoints.min(axis=0).astype(int)
        x_max, y_max = keypoints.max(axis=0).astype(int)
        
        # Add padding
        width = x_max - x_min
        height = y_max - y_min
        pad_x = int(width * padding)
        pad_y = int(height * padding)
        
        x_min = max(0, x_min - pad_x)
        y_min = max(0, y_min - pad_y)
        x_max = min(frame.shape[1], x_max + pad_x)
        y_max = min(frame.shape[0], y_max + pad_y)
        
        # Ensure minimum size
        if x_max - x_min < 10 or y_max - y_min < 10:
            return None
        
        roi = frame[y_min:y_max, x_min:x_max]
        return roi
    
    def draw_landmarks(
        self,
        frame: np.ndarray,
        hand_landmarks: object,
        handedness: str,
        confidence: float,
    ) -> np.ndarray:
        """
        Draw hand landmarks and skeleton on frame (custom manual drawing).
        
        Args:
            frame (np.ndarray): Input frame (BGR)
            hand_landmarks: MediaPipe hand landmarks object
            handedness (str): 'Left' or 'Right'
            confidence (float): Detection confidence score
            
        Returns:
            np.ndarray: Frame with drawn landmarks
        """
        if frame is None:
            return frame
        
        frame_annotated = frame.copy()
        height, width, _ = frame.shape
        
        # Extract keypoints
        keypoints = []
        for landmark in hand_landmarks.landmark:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            keypoints.append((x, y))
        
        if len(keypoints) < 21:
            return frame_annotated
        
        # Define hand connections (finger joints)
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),  # Index
            (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
            (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
            (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
        ]
        
        # Draw connections (lines) first - so they appear behind circles
        for start_idx, end_idx in connections:
            if start_idx < len(keypoints) and end_idx < len(keypoints):
                start_point = keypoints[start_idx]
                end_point = keypoints[end_idx]
                cv2.line(frame_annotated, start_point, end_point, (0, 255, 0), 2)
        
        # Draw keypoints (circles)
        for idx, (x, y) in enumerate(keypoints):
            # Different colors for different joints
            if idx == 0:
                color = (255, 0, 0)  # Wrist - Blue
                radius = 8
            elif idx in [4, 8, 12, 16, 20]:
                color = (0, 255, 0)  # Fingertips - Green
                radius = 6
            else:
                color = (255, 255, 0)  # Other joints - Cyan
                radius = 5
            
            cv2.circle(frame_annotated, (x, y), radius, color, -1)
            cv2.circle(frame_annotated, (x, y), radius, (255, 255, 255), 1)
        
        # Add handedness label
        label_color = (0, 255, 0) if handedness == 'Right' else (255, 0, 0)
        cv2.putText(
            frame_annotated,
            f"{handedness} ({confidence:.2f})",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            label_color,
            2,
        )
        
        return frame_annotated
    
    def draw_roi_box(
        self,
        frame: np.ndarray,
        keypoints: np.ndarray,
        padding: float = 0.2,
        color: Tuple[int, int, int] = (0, 255, 0),
    ) -> np.ndarray:
        """
        Draw ROI bounding box on frame.
        
        Args:
            frame (np.ndarray): Input frame
            keypoints (np.ndarray): Hand keypoints
            padding (float): Padding ratio
            color (Tuple): Box color (B, G, R)
            
        Returns:
            np.ndarray: Frame with drawn box
        """
        # Safety checks for None or invalid inputs
        if frame is None:
            return frame
        if keypoints is None or len(keypoints) < 21:
            return frame
        
        try:
            x_min, y_min = keypoints.min(axis=0).astype(int)
            x_max, y_max = keypoints.max(axis=0).astype(int)
            
            width = x_max - x_min
            height = y_max - y_min
            pad_x = int(width * padding)
            pad_y = int(height * padding)
            
            x_min = max(0, x_min - pad_x)
            y_min = max(0, y_min - pad_y)
            x_max = min(frame.shape[1], x_max + pad_x)
            y_max = min(frame.shape[0], y_max + pad_y)
            
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
        except Exception as e:
            logger.warning(f"Error drawing ROI box: {e}")
        
        return frame
    
    def close(self) -> None:
        """Clean up resources."""
        if self.hands:
            self.hands.close()
        logger.info("HandDetector closed")
