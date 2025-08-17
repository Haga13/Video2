#!/usr/bin/env python3
"""
Pure MediaPipe Multi-Person Pose Detection
Uses MediaPipe Face Detection + Pose Detection for multiple people
No YOLOv8 dependency - pure MediaPipe solution
"""

import cv2
import numpy as np
import time
import math
import mediapipe as mp

class MediaPipeMultiPersonDetector:
    def __init__(self, 
                 face_confidence=0.3,
                 pose_confidence=0.5,
                 max_people=8,
                 min_face_size=30):
        """
        Initialize Pure MediaPipe Multi-Person Detector
        
        Args:
            face_confidence: confidence threshold for face detection (lower = detect more distant people)
            pose_confidence: confidence threshold for pose landmarks
            max_people: maximum number of people to track simultaneously
            min_face_size: minimum face size in pixels (lower = more distant people)
        """
        self.face_confidence = face_confidence
        self.pose_confidence = pose_confidence
        self.max_people = max_people
        self.min_face_size = min_face_size
        
        # Initialize MediaPipe Face Detection
        print("Loading MediaPipe Face Detection...")
        try:
            self.mp_face_detection = mp.solutions.face_detection
            self.face_detector = self.mp_face_detection.FaceDetection(
                model_selection=1,  # 0 for close faces, 1 for faces within 5 meters
                min_detection_confidence=face_confidence
            )
            print("✅ MediaPipe Face Detection loaded")
        except Exception as e:
            print(f"❌ Error loading MediaPipe Face Detection: {e}")
            self.face_detector = None
        
        # Initialize MediaPipe Pose
        print("Loading MediaPipe Pose...")
        try:
            self.mp_pose = mp.solutions.pose
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles
            
            self.pose_detector = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,  # 0=light, 1=full, 2=heavy
                smooth_landmarks=True,
                enable_segmentation=False,
                min_detection_confidence=pose_confidence,
                min_tracking_confidence=0.5
            )
            print("✅ MediaPipe Pose loaded")
        except Exception as e:
            print(f"❌ Error loading MediaPipe Pose: {e}")
            self.pose_detector = None
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
        # Colors for different people
        self.person_colors = [
            (255, 0, 0),   # Red
            (0, 255, 0),   # Green
            (0, 0, 255),   # Blue
            (255, 255, 0), # Yellow
            (255, 0, 255), # Magenta
            (0, 255, 255), # Cyan
            (128, 0, 128), # Purple
            (255, 165, 0), # Orange
            (0, 128, 0),   # Dark Green
            (128, 128, 128) # Gray
        ]
        
        # Track people across frames for smoother detection
        self.previous_people = []
        self.person_id_counter = 0
    
    def detect_faces(self, image):
        """
        Detect faces to locate people in image
        
        Args:
            image: BGR image from OpenCV
            
        Returns:
            face_regions: list of face bounding boxes and info
        """
        if self.face_detector is None:
            return []
        
        try:
            # Convert BGR to RGB for MediaPipe
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            rgb_image.flags.writeable = False
            
            # Detect faces
            results = self.face_detector.process(rgb_image)
            face_regions = []
            
            if results.detections:
                h, w = image.shape[:2]
                
                for detection in results.detections:
                    # Get bounding box
                    bbox = detection.location_data.relative_bounding_box
                    
                    # Convert to pixel coordinates
                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    width = int(bbox.width * w)
                    height = int(bbox.height * h)
                    
                    # Filter by minimum face size
                    if width >= self.min_face_size and height >= self.min_face_size:
                        # Expand region for full body (faces are typically at top 1/8 of body)
                        body_height = height * 8  # Estimate full body height
                        body_width = width * 3    # Estimate body width
                        
                        # Calculate full body bounding box
                        body_x = max(0, x - width)  # Center horizontally around face
                        body_y = max(0, y - height // 2)  # Start slightly above face
                        body_x2 = min(w, body_x + body_width)
                        body_y2 = min(h, body_y + body_height)
                        
                        face_info = {
                            'face_bbox': [x, y, x + width, y + height],
                            'body_bbox': [body_x, body_y, body_x2, body_y2],
                            'confidence': detection.score[0],
                            'face_size': max(width, height),
                            'distance': self._estimate_distance_from_face(max(width, height))
                        }
                        
                        face_regions.append(face_info)
            
            # Sort by confidence and limit to max_people
            face_regions.sort(key=lambda x: x['confidence'], reverse=True)
            return face_regions[:self.max_people]
            
        except Exception as e:
            print(f"Face detection error: {e}")
            return []
    
    def detect_pose_for_person(self, image, person_region):
        """
        Detect pose for a specific person region using MediaPipe
        
        Args:
            image: full BGR image
            person_region: region info with body_bbox
            
        Returns:
            pose_results: MediaPipe pose results
        """
        if self.pose_detector is None:
            return None
        
        try:
            # Extract body bounding box
            x1, y1, x2, y2 = person_region['body_bbox']
            
            # Ensure valid crop region
            h, w = image.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 <= x1 or y2 <= y1:
                return None
            
            # Crop person region
            person_crop = image[y1:y2, x1:x2]
            
            if person_crop.size == 0:
                return None
            
            # Convert to RGB for MediaPipe
            rgb_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
            rgb_crop.flags.writeable = False
            
            # Run pose detection on cropped region
            results = self.pose_detector.process(rgb_crop)
            
            # Convert landmarks back to full image coordinates
            if results.pose_landmarks:
                crop_height, crop_width = person_crop.shape[:2]
                
                for landmark in results.pose_landmarks.landmark:
                    # Convert normalized coordinates to full image coordinates
                    landmark.x = (landmark.x * crop_width + x1) / w
                    landmark.y = (landmark.y * crop_height + y1) / h
            
            return results
            
        except Exception as e:
            print(f"Pose detection error for person: {e}")
            return None
    
    def detect_all_poses(self, image):
        """
        Detect poses for all people in image using face detection + pose detection
        
        Args:
            image: BGR image
            
        Returns:
            all_poses: list of pose data for each detected person
        """
        # First detect faces to locate people
        face_regions = self.detect_faces(image)
        all_poses = []
        
        # Then detect pose for each person region
        for i, face_region in enumerate(face_regions):
            pose_results = self.detect_pose_for_person(image, face_region)
            
            if pose_results and pose_results.pose_landmarks:
                pose_data = {
                    'person_id': i,
                    'face_bbox': face_region['face_bbox'],
                    'body_bbox': face_region['body_bbox'],
                    'face_confidence': face_region['confidence'],
                    'face_size': face_region['face_size'],
                    'distance': face_region['distance'],
                    'pose_landmarks': pose_results.pose_landmarks,
                    'color': self.person_colors[i % len(self.person_colors)]
                }
                all_poses.append(pose_data)
        
        return all_poses
    
    def draw_poses(self, image, all_poses, draw_skeleton=True, draw_landmarks=True, draw_bbox=True, draw_face_bbox=False):
        """
        Draw all detected poses on image
        
        Args:
            image: BGR image
            all_poses: list of pose data from detect_all_poses()
            draw_skeleton: whether to draw pose connections
            draw_landmarks: whether to draw landmark points
            draw_bbox: whether to draw body bounding boxes
            draw_face_bbox: whether to draw face bounding boxes
            
        Returns:
            annotated_image: image with pose annotations
        """
        annotated_image = image.copy()
        
        for pose_data in all_poses:
            person_id = pose_data['person_id']
            face_bbox = pose_data['face_bbox']
            body_bbox = pose_data['body_bbox']
            confidence = pose_data['face_confidence']
            distance = pose_data['distance']
            landmarks = pose_data['pose_landmarks']
            color = pose_data['color']
            
            # Draw body bounding box
            if draw_bbox:
                x1, y1, x2, y2 = body_bbox
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
                
                # Label with person info
                label = f'Person {person_id}: {confidence:.2f} (~{distance:.1f}m)'
                
                # Background for text
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(annotated_image, (x1, y1-25), (x1 + label_size[0], y1), color, -1)
                
                cv2.putText(annotated_image, label, (x1, y1-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Draw face bounding box (smaller, for debugging)
            if draw_face_bbox:
                fx1, fy1, fx2, fy2 = face_bbox
                cv2.rectangle(annotated_image, (fx1, fy1), (fx2, fy2), color, 1)
                cv2.putText(annotated_image, "FACE", (fx1, fy1-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
            
            # Draw pose skeleton
            if draw_skeleton and landmarks:
                # Use MediaPipe's drawing utilities with custom color
                landmark_drawing_spec = mp.solutions.drawing_utils.DrawingSpec(
                    color=color, thickness=2, circle_radius=2)
                connection_drawing_spec = mp.solutions.drawing_utils.DrawingSpec(
                    color=color, thickness=2)
                
                self.mp_drawing.draw_landmarks(
                    annotated_image, landmarks, self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=landmark_drawing_spec,
                    connection_drawing_spec=connection_drawing_spec)
            
            # Draw individual landmarks
            if draw_landmarks and landmarks:
                h, w = image.shape[:2]
                for landmark in landmarks.landmark:
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    cv2.circle(annotated_image, (x, y), 3, color, -1)
        
        return annotated_image
    
    def analyze_poses(self, all_poses, image_shape):
        """
        Analyze detected poses for useful information
        
        Args:
            all_poses: list of pose data
            image_shape: (height, width) of the image
            
        Returns:
            analysis_results: list of analysis for each pose
        """
        analysis_results = []
        h, w = image_shape[:2]
        
        for pose_data in all_poses:
            landmarks = pose_data['pose_landmarks']
            analysis = {
                'person_id': pose_data['person_id'],
                'distance': pose_data['distance'],
                'face_size': pose_data['face_size']
            }
            
            if landmarks:
                # Convert landmarks to pixel coordinates
                landmark_points = []
                for landmark in landmarks.landmark:
                    x = landmark.x * w
                    y = landmark.y * h
                    visibility = landmark.visibility
                    landmark_points.append([x, y, visibility])
                
                landmark_points = np.array(landmark_points)
                
                # Analyze pose orientation using shoulders
                left_shoulder = landmark_points[11]  # MediaPipe left shoulder
                right_shoulder = landmark_points[12] # MediaPipe right shoulder
                
                if left_shoulder[2] > 0.5 and right_shoulder[2] > 0.5:  # Both visible
                    shoulder_distance = abs(left_shoulder[0] - right_shoulder[0])
                    
                    if shoulder_distance > 60:
                        orientation = 'frontal'
                    elif shoulder_distance > 30:
                        orientation = 'angled'
                    else:
                        orientation = 'sideways'
                    
                    analysis['orientation'] = orientation
                    analysis['shoulder_distance'] = shoulder_distance
                
                # Calculate arm angles (if visible)
                try:
                    # Left arm: shoulder(11) -> elbow(13) -> wrist(15)
                    if all(landmark_points[i][2] > 0.5 for i in [11, 13, 15]):
                        angle = self._calculate_angle(
                            landmark_points[11][:2],
                            landmark_points[13][:2], 
                            landmark_points[15][:2]
                        )
                        analysis['left_arm_angle'] = angle
                    
                    # Right arm: shoulder(12) -> elbow(14) -> wrist(16)
                    if all(landmark_points[i][2] > 0.5 for i in [12, 14, 16]):
                        angle = self._calculate_angle(
                            landmark_points[12][:2],
                            landmark_points[14][:2],
                            landmark_points[16][:2]
                        )
                        analysis['right_arm_angle'] = angle
                        
                except Exception as e:
                    analysis['analysis_error'] = str(e)
                
                # Count visible landmarks
                visible_count = np.sum(landmark_points[:, 2] > 0.5)
                analysis['visible_landmarks'] = f"{visible_count}/33"
            
            analysis_results.append(analysis)
        
        return analysis_results
    
    def _calculate_angle(self, point1, point2, point3):
        """Calculate angle between three points in degrees"""
        vector1 = np.array([point1[0] - point2[0], point1[1] - point2[1]])
        vector2 = np.array([point3[0] - point2[0], point3[1] - point2[1]])
        
        norm1, norm2 = np.linalg.norm(vector1), np.linalg.norm(vector2)
        if norm1 == 0 or norm2 == 0:
            return 0
        
        cosine_angle = np.dot(vector1, vector2) / (norm1 * norm2)
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        angle = np.arccos(cosine_angle)
        
        return math.degrees(angle)
    
    def _estimate_distance_from_face(self, face_size_pixels):
        """Estimate distance based on face size in pixels"""
        if face_size_pixels == 0:
            return 0
        
        # Simplified distance estimation based on average face size
        # Average human face width is about 14cm
        focal_length_approx = 600
        real_face_size_meters = 0.14  # 14cm average face width
        distance = (real_face_size_meters * focal_length_approx) / face_size_pixels
        return round(distance, 1)
    
    def update_fps(self):
        """Update FPS counter"""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.fps_start_time >= 1.0:
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def set_detection_parameters(self, face_confidence=None, min_face_size=None, max_people=None):
        """Update detection parameters"""
        if face_confidence is not None:
            self.face_confidence = face_confidence
            # Update face detector with new confidence
            self.face_detector = self.mp_face_detection.FaceDetection(
                model_selection=1,
                min_detection_confidence=face_confidence
            )
            print(f"Set face confidence to {face_confidence}")
        
        if min_face_size is not None:
            self.min_face_size = min_face_size
            # Estimate max distance
            focal_length_approx = 600
            real_face_size_meters = 0.14
            max_distance = (real_face_size_meters * focal_length_approx) / min_face_size
            print(f"Set min face size to {min_face_size}px (~{max_distance:.1f}m max distance)")
        
        if max_people is not None:
            self.max_people = max_people
            print(f"Set max people to {max_people}")
    
    def get_detection_settings(self):
        """Get current detection settings"""
        # Estimate max distance from min face size
        focal_length_approx = 600
        real_face_size_meters = 0.14
        max_distance = (real_face_size_meters * focal_length_approx) / self.min_face_size
        
        return {
            'face_confidence': self.face_confidence,
            'pose_confidence': self.pose_confidence,
            'min_face_size': self.min_face_size,
            'max_people': self.max_people,
            'approximate_max_distance': round(max_distance, 1)
        }
    
    def draw_info_overlay(self, image, all_poses, analysis_results):
        """Draw information overlay"""
        overlay_image = image.copy()
        
        # FPS and detection info
        cv2.putText(overlay_image, f"FPS: {self.current_fps}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        cv2.putText(overlay_image, f"People: {len(all_poses)}/{self.max_people}", (10, 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Detection settings
        settings = self.get_detection_settings()
        cv2.putText(overlay_image, f"Max dist: ~{settings['approximate_max_distance']}m", (10, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.putText(overlay_image, f"Min face: {settings['min_face_size']}px", (10, 125),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Show analysis for detected people
        y_offset = 160
        for i, analysis in enumerate(analysis_results[:3]):  # Show first 3 people
            person_info = f"P{analysis['person_id']}: {analysis.get('orientation', 'unknown')}"
            if 'distance' in analysis:
                person_info += f" ~{analysis['distance']}m"
            
            color = self.person_colors[i % len(self.person_colors)]
            cv2.putText(overlay_image, person_info, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_offset += 25
        
        # Model info
        cv2.putText(overlay_image, "Pure MediaPipe Multi-Person", (10, image.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        
        return overlay_image

def main():
    """Main function to run pure MediaPipe multi-person pose detection"""
    print("=" * 70)
    print("🤸 PURE MEDIAPIPE MULTI-PERSON POSE DETECTION")
    print("=" * 70)
    print("Features:")
    print("  ✅ Multiple people detection (up to 8)")
    print("  ✅ Pure MediaPipe - no external dependencies")
    print("  ✅ Face detection for people location")
    print("  ✅ High-quality pose landmarks")
    print("  ✅ Color-coded people tracking")
    print("  ✅ Good distance detection")
    print("\nTechnology:")
    print("  👤 MediaPipe Face Detection (locates people)")
    print("  🤸 MediaPipe Pose Detection (analyzes each person)")
    print("\nControls:")
    print("  [SPACE] - Toggle landmarks")
    print("  [c] - Toggle skeleton connections")
    print("  [b] - Toggle body bounding boxes")
    print("  [f] - Toggle face bounding boxes")
    print("  [1-8] - Set max people (1-8)")
    print("  [+/-] - Increase/decrease face detection confidence")
    print("  [s] - Save screenshot")
    print("  [q/ESC] - Exit")
    print("=" * 70)
    
    # Initialize detector
    detector = MediaPipeMultiPersonDetector(
        face_confidence=0.3,      # Lower = detect more distant faces
        pose_confidence=0.5,      # Pose landmark confidence
        max_people=5,             # Track up to 5 people
        min_face_size=25          # Minimum face size in pixels (lower = more distant)
    )
    
    if detector.face_detector is None or detector.pose_detector is None:
        print("❌ Failed to load MediaPipe models. Please check installation.")
        return
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return
    
    print("✅ Camera and MediaPipe models initialized successfully")
    
    # Show initial settings
    settings = detector.get_detection_settings()
    print(f"📏 Initial settings:")
    print(f"   Max people: {settings['max_people']}")
    print(f"   Face confidence: {settings['face_confidence']}")
    print(f"   Max distance: ~{settings['approximate_max_distance']}m")
    print(f"   Min face size: {settings['min_face_size']}px")
    
    print("💡 Try multiple people in frame - works great with sideways poses!")
    print("🎯 Pure MediaPipe solution - no YOLOv8 needed!")
    
    # Control flags
    draw_landmarks = True
    draw_skeleton = True
    draw_bbox = True
    draw_face_bbox = False
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read frame")
                break
            
            # Flip frame for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Detect all poses
            all_poses = detector.detect_all_poses(frame)
            
            # Analyze poses
            analysis_results = detector.analyze_poses(all_poses, frame.shape)
            
            # Draw poses
            annotated_frame = detector.draw_poses(
                frame, all_poses, draw_skeleton, draw_landmarks, draw_bbox, draw_face_bbox)
            
            # Add info overlay
            final_frame = detector.draw_info_overlay(
                annotated_frame, all_poses, analysis_results)
            
            # Update FPS
            detector.update_fps()
            
            # Display frame
            cv2.imshow('Pure MediaPipe Multi-Person Pose Detection', final_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # 'q' or ESC
                break
            elif key == ord(' '):  # SPACE - toggle landmarks
                draw_landmarks = not draw_landmarks
                print(f"Landmarks: {'ON' if draw_landmarks else 'OFF'}")
            elif key == ord('c'):  # 'c' - toggle skeleton
                draw_skeleton = not draw_skeleton
                print(f"Skeleton: {'ON' if draw_skeleton else 'OFF'}")
            elif key == ord('b'):  # 'b' - toggle body bounding boxes
                draw_bbox = not draw_bbox
                print(f"Body boxes: {'ON' if draw_bbox else 'OFF'}")
            elif key == ord('f'):  # 'f' - toggle face bounding boxes
                draw_face_bbox = not draw_face_bbox
                print(f"Face boxes: {'ON' if draw_face_bbox else 'OFF'}")
            elif key in [ord(str(i)) for i in range(1, 9)]:  # 1-8 - set max people
                max_people = int(chr(key))
                detector.set_detection_parameters(max_people=max_people)
            elif key == ord('+') or key == ord('='):  # + - increase confidence
                new_conf = min(0.9, detector.face_confidence + 0.1)
                detector.set_detection_parameters(face_confidence=new_conf)
            elif key == ord('-'):  # - - decrease confidence
                new_conf = max(0.1, detector.face_confidence - 0.1)
                detector.set_detection_parameters(face_confidence=new_conf)
            elif key == ord('s'):  # 's' - save screenshot
                filename = f"mediapipe_pure_multipose_{int(time.time())}.jpg"
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
        print("\n💡 Install required packages:")
        print("   pip install opencv-python mediapipe numpy")
    finally:
        cv2.destroyAllWindows()