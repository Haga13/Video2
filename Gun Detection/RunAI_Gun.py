#!/usr/bin/env python3
"""
Kamera Capture Standalone
Script untuk menguji kamera dan melihat preview sebelum streaming
"""

import cv2
import sys
import time

class CameraPreview:
    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.cap = None
        
    def test_camera(self):
        """Test apakah kamera tersedia"""
        print(f"Testing kamera index {self.camera_index}...")
        
        self.cap = cv2.VideoCapture(self.camera_index)
        
        if not self.cap.isOpened():
            print(f"❌ Tidak dapat membuka kamera dengan index {self.camera_index}")
            return False
        
        # Test capture frame
        ret, frame = self.cap.read()
        if not ret:
            print(f"❌ Tidak dapat membaca frame dari kamera {self.camera_index}")
            return False
        
        # Get camera properties
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        print(f"✅ Kamera berhasil diinisialisasi!")
        print(f"   Resolusi: {width}x{height}")
        print(f"   FPS: {fps}")
        print(f"   Frame shape: {frame.shape}")
        
        return True
    
    def run_preview(self):
        """Jalankan preview kamera"""
        if not self.test_camera():
            return False
        
        print("\n" + "="*50)
        print("CAMERA PREVIEW")
        print("="*50)
        print("Tekan tombol berikut untuk kontrol:")
        print("  [SPACE] - Ambil screenshot")
        print("  [s]     - Simpan frame")
        print("  [q]     - Keluar")
        print("  [ESC]   - Keluar")
        print("="*50)
        
        # Setup window
        cv2.namedWindow('Camera Preview', cv2.WINDOW_NORMAL)
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Gagal membaca frame dari kamera")
                    break
                
                frame_count += 1
                
                # Calculate actual FPS
                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    actual_fps = 30 / elapsed if elapsed > 0 else 0
                    start_time = time.time()
                    
                    # Add FPS text to frame
                    cv2.putText(frame, f"FPS: {actual_fps:.1f}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Add frame counter
                cv2.putText(frame, f"Frame: {frame_count}", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Add timestamp
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                cv2.putText(frame, timestamp, (10, frame.shape[0] - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                # Show frame
                cv2.imshow('Camera Preview', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == 27:  # 'q' or ESC
                    break
                elif key == ord(' ') or key == ord('s'):  # SPACE or 's'
                    # Save screenshot
                    filename = f"camera_capture_{int(time.time())}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"📸 Screenshot disimpan: {filename}")
                
        except KeyboardInterrupt:
            print("\n⏹️  Preview dihentikan oleh user")
        
        finally:
            self.cleanup()
        
        return True
    
    def cleanup(self):
        """Cleanup resources"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("🔧 Camera resources released")

def scan_cameras():
    """Scan semua kamera yang tersedia"""
    print("🔍 Scanning untuk kamera yang tersedia...")
    available_cameras = []
    
    for i in range(10):  # Check first 10 indices
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                
                available_cameras.append({
                    'index': i,
                    'width': width,
                    'height': height,
                    'fps': fps
                })
                
                print(f"  📷 Kamera {i}: {width}x{height} @ {fps} FPS")
        cap.release()
    
    if not available_cameras:
        print("❌ Tidak ada kamera yang ditemukan")
        return None
    
    return available_cameras

def main():
    print("=" * 60)
    print("🎥 CAMERA TESTING & PREVIEW TOOL")
    print("=" * 60)
    
    # Scan cameras
    cameras = scan_cameras()
    
    if not cameras:
        print("\n💡 Tips:")
        print("  - Pastikan kamera terhubung dengan benar")
        print("  - Tutup aplikasi lain yang menggunakan kamera")
        print("  - Coba restart komputer jika perlu")
        return
    
    # Choose camera
    if len(cameras) == 1:
        camera_index = cameras[0]['index']
        print(f"\n🎯 Menggunakan kamera {camera_index} (satu-satunya yang tersedia)")
    else:
        print(f"\n📋 Ditemukan {len(cameras)} kamera:")
        for cam in cameras:
            print(f"  {cam['index']}: {cam['width']}x{cam['height']} @ {cam['fps']} FPS")
        
        while True:
            try:
                choice = input(f"\nPilih kamera (0-{len(cameras)-1}): ").strip()
                camera_index = int(choice)
                
                if any(cam['index'] == camera_index for cam in cameras):
                    break
                else:
                    print("❌ Index kamera tidak valid!")
            except ValueError:
                print("❌ Masukkan angka yang valid!")
    
    # Run preview
    preview = CameraPreview(camera_index)
    
    if preview.run_preview():
        print(f"\n✅ Preview kamera {camera_index} selesai")
    else:
        print(f"\n❌ Gagal menjalankan preview kamera {camera_index}")
    
    print("\n💡 Jika kamera bekerja dengan baik, Anda dapat menjalankan:")
    print("     python video_server.py")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️  Program dihentikan oleh user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        cv2.destroyAllWindows()
