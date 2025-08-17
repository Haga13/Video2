#!/usr/bin/env python3
"""
Video Client Viewer
Client untuk melihat video stream dari server dengan latency rendah
"""

import cv2
import requests
import numpy as np
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import queue

class VideoClient:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Video Stream Client - Low Latency")
        self.root.geometry("900x700")
        
        # Variables
        self.server_url = tk.StringVar(value="http://192.168.1.100:5000")
        self.is_connected = False
        self.stream_thread = None
        self.frame_queue = queue.Queue(maxsize=2)
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup user interface"""
        # Frame utama
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Konfigurasi grid
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # Header
        header_frame = ttk.LabelFrame(main_frame, text="Koneksi Server", padding="10")
        header_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        header_frame.columnconfigure(1, weight=1)
        
        ttk.Label(header_frame, text="Server URL:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        
        url_entry = ttk.Entry(header_frame, textvariable=self.server_url, width=40)
        url_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 5))
        
        self.connect_btn = ttk.Button(header_frame, text="Connect", command=self.toggle_connection)
        self.connect_btn.grid(row=0, column=2, padx=(5, 0))
        
        # Status
        self.status_label = ttk.Label(header_frame, text="Status: Disconnected", foreground="red")
        self.status_label.grid(row=1, column=0, columnspan=3, sticky=tk.W, pady=(5, 0))
        
        # Controls
        control_frame = ttk.LabelFrame(main_frame, text="Kontrol", padding="10")
        control_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Button(control_frame, text="Fullscreen", command=self.toggle_fullscreen).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(control_frame, text="Screenshot", command=self.take_screenshot).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(control_frame, text="Refresh", command=self.refresh_stream).pack(side=tk.LEFT)
        
        # Video display
        video_frame = ttk.LabelFrame(main_frame, text="Video Stream", padding="10")
        video_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        video_frame.columnconfigure(0, weight=1)
        video_frame.rowconfigure(0, weight=1)
        
        # Canvas untuk video
        self.canvas = tk.Canvas(video_frame, bg='black', width=640, height=480)
        self.canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Scrollbars
        v_scrollbar = ttk.Scrollbar(video_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        v_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.canvas.configure(yscrollcommand=v_scrollbar.set)
        
        h_scrollbar = ttk.Scrollbar(video_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        h_scrollbar.grid(row=1, column=0, sticky=(tk.W, tk.E))
        self.canvas.configure(xscrollcommand=h_scrollbar.set)
        
        # Info panel
        info_frame = ttk.LabelFrame(main_frame, text="Informasi", padding="10")
        info_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        info_text = """
Cara Penggunaan:
1. Masukkan URL server (contoh: http://192.168.1.100:5000)
2. Klik 'Connect' untuk mulai streaming
3. Gunakan tombol kontrol untuk mengelola tampilan
4. Klik 'Fullscreen' untuk tampilan penuh
5. Gunakan 'Screenshot' untuk menyimpan gambar

Tips:
- Pastikan perangkat terhubung ke jaringan yang sama
- Gunakan IP address yang benar dari server
- Tutup aplikasi lain yang menggunakan bandwidth tinggi
        """
        
        info_label = ttk.Label(info_frame, text=info_text.strip(), justify=tk.LEFT)
        info_label.pack()
        
    def toggle_connection(self):
        """Toggle koneksi ke server"""
        if not self.is_connected:
            self.connect_to_server()
        else:
            self.disconnect_from_server()
    
    def connect_to_server(self):
        """Koneksi ke server streaming"""
        server_url = self.server_url.get().strip()
        
        if not server_url:
            messagebox.showerror("Error", "Masukkan URL server!")
            return
        
        if not server_url.startswith('http'):
            server_url = 'http://' + server_url
            self.server_url.set(server_url)
        
        try:
            # Test koneksi
            response = requests.get(f"{server_url}/status", timeout=5)
            if response.status_code == 200:
                self.is_connected = True
                self.status_label.config(text="Status: Connected", foreground="green")
                self.connect_btn.config(text="Disconnect")
                
                # Mulai streaming thread
                self.stream_thread = threading.Thread(target=self.stream_worker, daemon=True)
                self.stream_thread.start()
                
                # Mulai update display
                self.update_display()
                
            else:
                raise Exception("Server tidak merespons dengan benar")
                
        except Exception as e:
            messagebox.showerror("Connection Error", f"Tidak dapat terhubung ke server:\n{str(e)}")
            self.status_label.config(text="Status: Connection Failed", foreground="red")
    
    def disconnect_from_server(self):
        """Disconnect dari server"""
        self.is_connected = False
        self.status_label.config(text="Status: Disconnected", foreground="red")
        self.connect_btn.config(text="Connect")
        
        # Clear canvas
        self.canvas.delete("all")
        self.canvas.create_text(320, 240, text="Disconnected", fill="white", font=("Arial", 20))
    
    def stream_worker(self):
        """Worker thread untuk streaming video"""
        server_url = self.server_url.get().strip()
        video_url = f"{server_url}/video_feed"
        
        try:
            # Setup stream
            stream = requests.get(video_url, stream=True, timeout=10)
            stream.raise_for_status()
            
            bytes_data = b''
            
            for chunk in stream.iter_content(chunk_size=1024):
                if not self.is_connected:
                    break
                    
                bytes_data += chunk
                
                # Cari boundary frame
                start = bytes_data.find(b'\xff\xd8')  # JPEG start
                end = bytes_data.find(b'\xff\xd9')    # JPEG end
                
                if start != -1 and end != -1:
                    # Extract JPEG frame
                    jpg_data = bytes_data[start:end+2]
                    bytes_data = bytes_data[end+2:]
                    
                    try:
                        # Decode image
                        nparr = np.frombuffer(jpg_data, np.uint8)
                        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        
                        if frame is not None:
                            # Clear queue untuk latency rendah
                            while not self.frame_queue.empty():
                                try:
                                    self.frame_queue.get_nowait()
                                except queue.Empty:
                                    break
                            
                            try:
                                self.frame_queue.put_nowait(frame)
                            except queue.Full:
                                pass
                                
                    except Exception as e:
                        print(f"Error decoding frame: {e}")
                        continue
                        
        except Exception as e:
            if self.is_connected:
                print(f"Stream error: {e}")
                self.root.after(0, lambda: messagebox.showerror("Stream Error", f"Error dalam streaming:\n{str(e)}"))
                self.root.after(0, self.disconnect_from_server)
    
    def update_display(self):
        """Update display dengan frame terbaru"""
        if not self.is_connected:
            return
        
        try:
            frame = self.frame_queue.get_nowait()
            
            # Convert BGR ke RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert ke PIL Image
            image = Image.fromarray(frame_rgb)
            
            # Resize untuk fit canvas
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            if canvas_width > 1 and canvas_height > 1:
                # Calculate aspect ratio
                img_width, img_height = image.size
                ratio = min(canvas_width/img_width, canvas_height/img_height)
                
                new_width = int(img_width * ratio)
                new_height = int(img_height * ratio)
                
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Convert ke PhotoImage
            photo = ImageTk.PhotoImage(image)
            
            # Update canvas
            self.canvas.delete("all")
            self.canvas.create_image(canvas_width//2, canvas_height//2, image=photo)
            self.canvas.image = photo  # Keep reference
            
        except queue.Empty:
            pass
        except Exception as e:
            print(f"Display error: {e}")
        
        # Schedule next update
        if self.is_connected:
            self.root.after(33, self.update_display)  # ~30 FPS
    
    def toggle_fullscreen(self):
        """Toggle fullscreen mode"""
        current_state = self.root.attributes('-fullscreen')
        self.root.attributes('-fullscreen', not current_state)
        
        if not current_state:
            # Entering fullscreen
            self.root.bind('<Escape>', lambda e: self.root.attributes('-fullscreen', False))
        else:
            # Exiting fullscreen
            self.root.unbind('<Escape>')
    
    def take_screenshot(self):
        """Ambil screenshot dari frame saat ini"""
        try:
            frame = self.frame_queue.get_nowait()
            
            # Save screenshot
            timestamp = int(time.time())
            filename = f"screenshot_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            
            messagebox.showinfo("Screenshot", f"Screenshot disimpan sebagai: {filename}")
            
        except queue.Empty:
            messagebox.showwarning("Screenshot", "Tidak ada frame untuk di-screenshot")
        except Exception as e:
            messagebox.showerror("Screenshot Error", f"Error saat menyimpan screenshot:\n{str(e)}")
    
    def refresh_stream(self):
        """Refresh stream"""
        if self.is_connected:
            self.disconnect_from_server()
            time.sleep(0.5)
            self.connect_to_server()
    
    def run(self):
        """Jalankan aplikasi"""
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            pass
        finally:
            self.is_connected = False

if __name__ == '__main__':
    try:
        print("Starting Video Stream Client...")
        print("Pastikan server streaming sudah berjalan di jaringan yang sama.")
        
        client = VideoClient()
        client.run()
        
    except Exception as e:
        print(f"Error: {e}")
        input("Press Enter to exit...")
