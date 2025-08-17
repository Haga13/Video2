#!/usr/bin/env python3
"""
Pose Detection Test System
Standalone script untuk testing pose detection menggunakan MediaPipe
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import math

class PoseDetector:
    def __init__(self, 
                 static_image_mode=False,
                 model_complexity=1,
                 smooth_landmarks=True,
                 enable_segmentation=False,
                 smooth_segmentation=True,
                 min_detection_confidence=0.7,
                 min_tracking_confidence=0.5):
        """
        Initialize MediaPipe Pose Detection
        
        Parameters:
        - model_complexity: 0 (light), 1 (full), 2 (heavy) - balance between speed and accuracy
        - min_detection_confidence: minimum confidence for pose detection
        - min_tracking_confidence: minimum confidence for pose tracking
        """
        
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            enable_segmentation=enable_segmentation,
            smooth_segmentation=smooth_segmentation,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
    def detect_pose(self, image):
        """
        Detect pose in image and return results
        
        Args:
            image: BGR image from OpenCV
            
        Returns:
            results: MediaPipe pose results
            annotated_image: Image with pose landmarks drawn
        """
        # Convert BGR to RGB (MediaPipe uses RGB)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb_image.flags.writeable = False
        
        # Perform pose detection
        results = self.pose.process(rgb_image)
        
        # Convert back to BGR for OpenCV
        rgb_image.flags.writeable = True
        annotated_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        
        return results, annotated_image
    
    def draw_landmarks(self, image, results, draw_connections=True, draw_landmarks=True):
        """
        Draw pose landmarks and connections on image
        
        Args:
            image: BGR image
            results: MediaPipe pose results
            draw_connections: whether to draw pose connections
            draw_landmarks: whether to draw individual landmarks
            
        Returns:
            annotated_image: Image with pose annotations
        """
        annotated_image = image.copy()
        
        if results.pose_landmarks:
            if draw_connections:
                # Draw pose connections with custom style
                self.mp_drawing.draw_landmarks(
                    annotated_image,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())
            
            if draw_landmarks:
                # Draw individual landmarks
                self.mp_drawing.draw_landmarks(
                    annotated_image,
                    results.pose_landmarks,
                    None,
                    landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                        color=(0, 255, 0), thickness=2, circle_radius=2))
        
        return annotated_image
    
    def get_landmark_coordinates(self, results, image_width, image_height):
        """
        Extract landmark coordinates in pixel values
        
        Args:
            results: MediaPipe pose results
            image_width: width of the image
            image_height: height of the image
            
        Returns:
            landmarks: dictionary of landmark coordinates
        """
        landmarks = {}
        
        if results.pose_landmarks:
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                # Convert normalized coordinates to pixels
                x = int(landmark.x * image_width)
                y = int(landmark.y * image_height)
                z = landmark.z  # Relative depth
                visibility = landmark.visibility
                
                # Get landmark name from MediaPipe
                landmark_name = self.mp_pose.PoseLandmark(idx).name
                
                landmarks[landmark_name] = {
                    'x': x,
                    'y': y,
                    'z': z,
                    'visibility': visibility
                }
        
        return landmarks
    
    def calculate_angle(self, point1, point2, point3):
        """
        Calculate angle between three points
        Useful for analyzing poses (e.g., arm bend, leg bend)
        
        Args:
            point1, point2, point3: tuples of (x, y) coordinates
            
        Returns:
            angle: angle in degrees
        """
        # Calculate vectors
        vector1 = np.array([point1[0] - point2[0], point1[1] - point2[1]])
        vector2 = np.array([point3[0] - point2[0], point3[1] - point2[1]])
        
        # Calculate angle
        cosine_angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
        angle = np.arccos(cosine_angle)
        
        return math.degrees(angle)
    
    def analyze_pose(self, landmarks):
        """
        Analyze pose and extract useful information
        
        Args:
            landmarks: landmark coordinates dictionary
            
        Returns:
            analysis: dictionary with pose analysis
        """
        analysis = {}
        
        if not landmarks:
            return analysis
        
        # Check if key landmarks are visible
        key_landmarks = ['LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 'RIGHT_ELBOW',
                        'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_HIP', 'RIGHT_HIP',
                        'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE', 'RIGHT_ANKLE']
        
        # Calculate some basic angles if landmarks are available
        try:
            # Left arm angle (shoulder-elbow-wrist)
            if all(lm in landmarks for lm in ['LEFT_SHOULDER', 'LEFT_ELBOW', 'LEFT_WRIST']):
                left_shoulder = (landmarks['LEFT_SHOULDER']['x'], landmarks['LEFT_SHOULDER']['y'])
                left_elbow = (landmarks['LEFT_ELBOW']['x'], landmarks['LEFT_ELBOW']['y'])
                left_wrist = (landmarks['LEFT_WRIST']['x'], landmarks['LEFT_WRIST']['y'])
                analysis['left_arm_angle'] = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
            
            # Right arm angle
            if all(lm in landmarks for lm in ['RIGHT_SHOULDER', 'RIGHT_ELBOW', 'RIGHT_WRIST']):
                right_shoulder = (landmarks['RIGHT_SHOULDER']['x'], landmarks['RIGHT_SHOULDER']['y'])
                right_elbow = (landmarks['RIGHT_ELBOW']['x'], landmarks['RIGHT_ELBOW']['y'])
                right_wrist = (landmarks['RIGHT_WRIST']['x'], landmarks['RIGHT_WRIST']['y'])
                analysis['right_arm_angle'] = self.calculate_angle(right_shoulder, right_elbow, right_wrist)
            
            # Body posture - shoulder alignment
            if 'LEFT_SHOULDER' in landmarks and 'RIGHT_SHOULDER' in landmarks:
                left_shoulder_y = landmarks['LEFT_SHOULDER']['y']
                right_shoulder_y = landmarks['RIGHT_SHOULDER']['y']
                shoulder_tilt = abs(left_shoulder_y - right_shoulder_y)
                analysis['shoulder_tilt'] = shoulder_tilt
                analysis['posture_status'] = 'good' if shoulder_tilt < 20 else 'tilted'
        
        except Exception as e:
            print(f"Error in pose analysis: {e}")
        
        return analysis
    
    def update_fps(self):
        """Update FPS counter"""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.fps_start_time >= 1.0:
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def draw_info_overlay(self, image, landmarks, analysis):
        """
        Draw information overlay on image
        
        Args:
            image: BGR image
            landmarks: landmark coordinates
            analysis: pose analysis results
            
        Returns:
            image with overlay
        """
        overlay_image = image.copy()
        
        # FPS counter
        cv2.putText(overlay_image, f"FPS: {self.current_fps}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Number of detected landmarks
        landmark_count = len(landmarks) if landmarks else 0
        cv2.putText(overlay_image, f"Landmarks: {landmark_count}/33", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Pose analysis info
        y_offset = 90
        if analysis:
            for key, value in analysis.items():
                if isinstance(value, float):
                    text = f"{key}: {value:.1f}"
                else:
                    text = f"{key}: {value}"
                
                cv2.putText(overlay_image, text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                y_offset += 25
        
        # Detection status
        status_text = "POSE DETECTED" if landmarks else "NO POSE"
        status_color = (0, 255, 0) if landmarks else (0, 0, 255)
        cv2.putText(overlay_image, status_text, (10, image.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        return overlay_image

def test_pose_detection():
    """Test pose detection with webcam"""
    print("=" * 60)
    print("🤸 POSE DETECTION TEST SYSTEM")
    print("=" * 60)
    print("Controls:")
    print("  [SPACE] - Toggle landmark drawing")
    print("  [c]     - Toggle connections drawing")
    print("  [s]     - Save screenshot")
    print("  [q/ESC] - Exit")
    print("=" * 60)
    
    # Initialize pose detector
    pose_detector = PoseDetector(
        model_complexity=1,  # Balance between speed and accuracy
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return
    
    # Control flags
    draw_landmarks = True
    draw_connections = True
    
    print("✅ Pose detection started! Move around to test detection...")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read frame")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Detect pose
            results, processed_frame = pose_detector.detect_pose(frame)
            
            # Draw pose landmarks
            if results.pose_landmarks:
                processed_frame = pose_detector.draw_landmarks(
                    processed_frame, results, draw_connections, draw_landmarks)
            
            # Get landmark coordinates
            landmarks = pose_detector.get_landmark_coordinates(
                results, frame.shape[1], frame.shape[0])
            
            # Analyze pose
            analysis = pose_detector.analyze_pose(landmarks)
            
            # Draw info overlay
            final_frame = pose_detector.draw_info_overlay(processed_frame, landmarks, analysis)
            
            # Update FPS
            pose_detector.update_fps()
            
            # Display frame
            cv2.imshow('Pose Detection Test', final_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # 'q' or ESC
                break
            elif key == ord(' '):  # SPACE - toggle landmarks
                draw_landmarks = not draw_landmarks
                print(f"Landmarks drawing: {'ON' if draw_landmarks else 'OFF'}")
            elif key == ord('c'):  # 'c' - toggle connections
                draw_connections = not draw_connections
                print(f"Connections drawing: {'ON' if draw_connections else 'OFF'}")
            elif key == ord('s'):  # 's' - save screenshot
                filename = f"pose_detection_{int(time.time())}.jpg"
                cv2.imwrite(filename, final_frame)
                print(f"📸 Screenshot saved: {filename}")
    
    except KeyboardInterrupt:
        print("\n⏹️ Test stopped by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("🔧 Resources released")

if __name__ == '__main__':
    try:
        test_pose_detection()
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 Make sure you have the required packages installed:")
        print("   pip install opencv-python mediapipe numpy")
    finally:
        cv2.destroyAllWindows()