#!/usr/bin/env python3
"""
YOLOv8 Pose Detection - Clean Interface
Detects people at any distance without size restrictions
"""

import cv2
import numpy as np
import time
import math
from ultralytics import YOLO

class YOLOv8PoseDetector:
    def __init__(self, model_size='n', confidence_threshold=0.5, input_size=640):
        """
        Initialize YOLOv8 Pose detector
        
        Args:
            model_size: 'n' (nano/fastest), 's' (small), 'm' (medium), 'l' (large), 'x' (extra large)
            confidence_threshold: minimum confidence for pose detection (0.0-1.0)
            input_size: input image size for the model (higher = detects smaller/distant objects)
                       Options: 320, 416, 512, 640, 832, 1024, 1280
        """
        self.confidence_threshold = confidence_threshold
        self.input_size = input_size
        
        # Load YOLOv8 pose model
        model_name = f"yolov8{model_size}-pose.pt"
        print(f"Loading YOLOv8 Pose model: {model_name}")
        print("Note: Model will be downloaded automatically on first use")
        
        try:
            self.model = YOLO(model_name)
            print("✅ YOLOv8 Pose model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading YOLOv8 model: {e}")
            self.model = None
            return
        
        # COCO pose keypoints (17 points)
        self.keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        
        # Skeleton connections for drawing (COCO format)
        self.skeleton_connections = [
            # Face
            [0, 1], [0, 2], [1, 3], [2, 4],
            # Arms
            [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
            # Body
            [5, 11], [6, 12], [11, 12],
            # Legs
            [11, 13], [13, 15], [12, 14], [14, 16]
        ]
        
        # Colors for different body parts
        self.colors = {
            'face': (255, 255, 0),      # Yellow
            'arms': (0, 255, 0),        # Green
            'body': (255, 0, 0),        # Red
            'legs': (0, 0, 255),        # Blue
            'keypoints': (0, 255, 255)  # Cyan
        }
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
    
    def detect_poses(self, image):
        """
        Detect poses in image using YOLOv8
        
        Args:
            image: BGR image from OpenCV
            
        Returns:
            poses: list of detected poses with keypoints and metadata
        """
        if self.model is None:
            return []
        
        try:
            # Run YOLOv8 inference with custom input size for better distance detection
            results = self.model(image, imgsz=self.input_size, verbose=False)
            poses = []
            
            for result in results:
                if result.keypoints is not None:
                    keypoints_data = result.keypoints.data  # Shape: [num_people, 17, 3]
                    boxes_data = result.boxes.data if result.boxes is not None else None
                    
                    # Process each detected person
                    for i, person_keypoints in enumerate(keypoints_data):
                        # Get confidence score from bounding box
                        bbox_confidence = boxes_data[i][4].item() if boxes_data is not None and i < len(boxes_data) else 1.0
                        
                        if bbox_confidence >= self.confidence_threshold:
                            # Convert keypoints to numpy array
                            kpts = person_keypoints.cpu().numpy()  # [17, 3] - (x, y, visibility)
                            
                            # Get bounding box if available
                            bbox = boxes_data[i][:4].cpu().numpy() if boxes_data is not None and i < len(boxes_data) else None
                            
                            # No size filtering - accept all detections regardless of distance
                            if bbox is not None:
                                bbox_width = bbox[2] - bbox[0]
                                bbox_height = bbox[3] - bbox[1]
                                person_size = max(bbox_width, bbox_height)
                            else:
                                person_size = 0
                            
                            pose_data = {
                                'person_id': i,
                                'keypoints': kpts,
                                'bbox_confidence': bbox_confidence,
                                'bbox': bbox,
                                'person_size': person_size,
                                'visible_keypoints': np.sum(kpts[:, 2] > 0.5)
                            }
                            
                            poses.append(pose_data)
            
            return poses
            
        except Exception as e:
            print(f"Detection error: {e}")
            return []
    
    def draw_poses(self, image, poses, draw_skeleton=True, draw_keypoints=True):
        """
        Draw detected poses on image
        
        Args:
            image: BGR image
            poses: list of pose data from detect_poses()
            draw_skeleton: whether to draw skeleton connections
            draw_keypoints: whether to draw individual keypoints
            
        Returns:
            annotated_image: image with pose annotations
        """
        annotated_image = image.copy()
        
        for pose in poses:
            keypoints = pose['keypoints']
            
            # Draw skeleton connections
            if draw_skeleton:
                for connection in self.skeleton_connections:
                    kpt1_idx, kpt2_idx = connection
                    
                    # Check if both keypoints are visible
                    if (keypoints[kpt1_idx][2] > 0.5 and keypoints[kpt2_idx][2] > 0.5):
                        pt1 = (int(keypoints[kpt1_idx][0]), int(keypoints[kpt1_idx][1]))
                        pt2 = (int(keypoints[kpt2_idx][0]), int(keypoints[kpt2_idx][1]))
                        
                        # Choose color based on body part
                        if kpt1_idx <= 4 or kpt2_idx <= 4:  # Face
                            color = self.colors['face']
                        elif kpt1_idx <= 10 or kpt2_idx <= 10:  # Arms
                            color = self.colors['arms']
                        elif kpt1_idx <= 12 or kpt2_idx <= 12:  # Body
                            color = self.colors['body']
                        else:  # Legs
                            color = self.colors['legs']
                        
                        cv2.line(annotated_image, pt1, pt2, color, 2)
            
            # Draw keypoints
            if draw_keypoints:
                for i, (x, y, visibility) in enumerate(keypoints):
                    if visibility > 0.5:  # Only draw visible keypoints
                        # Choose color and size based on keypoint type
                        if i <= 4:  # Face keypoints
                            color = self.colors['face']
                            radius = 3
                        else:  # Body keypoints
                            color = self.colors['keypoints']
                            radius = 4
                        
                        cv2.circle(annotated_image, (int(x), int(y)), radius, color, -1)
        
        return annotated_image
    
    def analyze_poses(self, poses):
        """
        Analyze detected poses for useful information
        
        Args:
            poses: list of pose data
            
        Returns:
            analysis: list of analysis data for each pose
        """
        analysis_results = []
        
        for pose in poses:
            keypoints = pose['keypoints']
            analysis = {'person_id': pose['person_id']}
            
            try:
                # Determine pose orientation (frontal vs sideways)
                left_shoulder = keypoints[5]   # left_shoulder
                right_shoulder = keypoints[6]  # right_shoulder
                
                if left_shoulder[2] > 0.5 and right_shoulder[2] > 0.5:
                    shoulder_distance = abs(left_shoulder[0] - right_shoulder[0])
                    
                    if shoulder_distance > 60:
                        orientation = 'frontal'
                    elif shoulder_distance > 30:
                        orientation = 'angled'
                    else:
                        orientation = 'sideways'
                    
                    analysis['orientation'] = orientation
                    analysis['shoulder_distance'] = shoulder_distance
                
                # Calculate arm angles
                self._calculate_arm_angles(keypoints, analysis)
                
                # Analyze posture
                self._analyze_posture(keypoints, analysis)
                
                # Count visible body parts
                face_visible = np.sum(keypoints[:5, 2] > 0.5)
                arms_visible = np.sum(keypoints[5:11, 2] > 0.5)
                legs_visible = np.sum(keypoints[11:, 2] > 0.5)
                
                analysis['visibility'] = {
                    'face': face_visible,
                    'arms': arms_visible,
                    'legs': legs_visible,
                    'total': pose['visible_keypoints']
                }
                
            except Exception as e:
                analysis['error'] = str(e)
            
            analysis_results.append(analysis)
        
        return analysis_results
    
    def _calculate_arm_angles(self, keypoints, analysis):
        """Calculate arm bend angles"""
        # Left arm angle (shoulder-elbow-wrist)
        if all(keypoints[i][2] > 0.5 for i in [5, 7, 9]):  # left shoulder, elbow, wrist
            angle = self._calculate_angle(
                keypoints[5][:2],  # shoulder
                keypoints[7][:2],  # elbow
                keypoints[9][:2]   # wrist
            )
            analysis['left_arm_angle'] = angle
        
        # Right arm angle
        if all(keypoints[i][2] > 0.5 for i in [6, 8, 10]):  # right shoulder, elbow, wrist
            angle = self._calculate_angle(
                keypoints[6][:2],  # shoulder
                keypoints[8][:2],  # elbow
                keypoints[10][:2]  # wrist
            )
            analysis['right_arm_angle'] = angle
    
    def _analyze_posture(self, keypoints, analysis):
        """Analyze overall posture"""
        # Shoulder level (tilt analysis)
        if keypoints[5][2] > 0.5 and keypoints[6][2] > 0.5:  # both shoulders visible
            shoulder_tilt = abs(keypoints[5][1] - keypoints[6][1])
            analysis['shoulder_tilt'] = shoulder_tilt
            analysis['posture_status'] = 'good' if shoulder_tilt < 25 else 'tilted'
        
        # Body lean (hip vs shoulder alignment)
        if all(keypoints[i][2] > 0.5 for i in [5, 6, 11, 12]):  # shoulders and hips
            shoulder_center = ((keypoints[5][0] + keypoints[6][0]) / 2,
                              (keypoints[5][1] + keypoints[6][1]) / 2)
            hip_center = ((keypoints[11][0] + keypoints[12][0]) / 2,
                         (keypoints[11][1] + keypoints[12][1]) / 2)
            
            body_lean = abs(shoulder_center[0] - hip_center[0])
            analysis['body_lean'] = body_lean
            analysis['lean_status'] = 'straight' if body_lean < 30 else 'leaning'
    
    def _calculate_angle(self, point1, point2, point3):
        """Calculate angle between three points (in degrees)"""
        vector1 = np.array([point1[0] - point2[0], point1[1] - point2[1]])
        vector2 = np.array([point3[0] - point2[0], point3[1] - point2[1]])
        
        # Avoid division by zero
        norm1, norm2 = np.linalg.norm(vector1), np.linalg.norm(vector2)
        if norm1 == 0 or norm2 == 0:
            return 0
        
        cosine_angle = np.dot(vector1, vector2) / (norm1 * norm2)
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)  # Handle numerical errors
        angle = np.arccos(cosine_angle)
        
        return math.degrees(angle)
    
    def update_fps(self):
        """Update FPS counter"""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.fps_start_time >= 1.0:
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def draw_simple_overlay(self, image, poses):
        """Draw simple information overlay"""
        overlay_image = image.copy()
        
        # Only show FPS and people count
        cv2.putText(overlay_image, f"FPS: {self.current_fps}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        cv2.putText(overlay_image, f"People detected: {len(poses)}", (10, 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        return overlay_image

def main():
    """Main function to run YOLOv8 pose detection"""
    print("=" * 60)
    print("🤸 YOLOv8 POSE DETECTION - CLEAN INTERFACE")
    print("=" * 60)
    print("Features:")
    print("  ✅ Detects people at any distance")
    print("  ✅ Multiple person detection")
    print("  ✅ No size restrictions")
    print("  ✅ Clean interface (no bounding boxes)")
    print("  ✅ Color-coded skeleton drawing")
    print("\nControls:")
    print("  [SPACE] - Toggle keypoints")
    print("  [c] - Toggle skeleton connections")
    print("  [s] - Save screenshot")
    print("  [q/ESC] - Exit")
    print("=" * 60)
    
    # Initialize detector without distance restrictions
    pose_detector = YOLOv8PoseDetector(
        model_size='n',          # Use nano model for speed
        confidence_threshold=0.5, # Standard confidence threshold
        input_size=640           # Standard input size
    )
    
    if pose_detector.model is None:
        print("❌ Failed to load YOLOv8 model. Please check your installation.")
        return
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return
    
    print("✅ Camera initialized successfully")
    print("💡 Detects people at any distance - no size restrictions!")
    
    # Control flags
    draw_keypoints = True
    draw_skeleton = True
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read frame")
                break
            
            # Flip frame for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Detect poses
            poses = pose_detector.detect_poses(frame)
            
            # Draw poses on frame (no bounding boxes)
            annotated_frame = pose_detector.draw_poses(
                frame, poses, draw_skeleton, draw_keypoints)
            
            # Add simple overlay
            final_frame = pose_detector.draw_simple_overlay(annotated_frame, poses)
            
            # Update FPS counter
            pose_detector.update_fps()
            
            # Display frame
            cv2.imshow('YOLOv8 Pose Detection - Clean', final_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # 'q' or ESC
                break
            elif key == ord(' '):  # SPACE - toggle keypoints
                draw_keypoints = not draw_keypoints
                print(f"Keypoints: {'ON' if draw_keypoints else 'OFF'}")
            elif key == ord('c'):  # 'c' - toggle skeleton
                draw_skeleton = not draw_skeleton
                print(f"Skeleton: {'ON' if draw_skeleton else 'OFF'}")
            elif key == ord('s'):  # 's' - save screenshot
                filename = f"yolov8_pose_clean_{int(time.time())}.jpg"
                cv2.imwrite(filename, final_frame)
                print(f"📸 Screenshot saved: {filename}")
    
    except KeyboardInterrupt:
        print("\n⏹️ Stopped by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("🔧 Resources released")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 Make sure you have the required packages:")
        print("   pip install ultralytics opencv-python numpy")
    finally:
        cv2.destroyAllWindows()