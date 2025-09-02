import cv2

def check_camera_indexes(max_index=10):
    """
    Mengecek kamera yang tersedia di sistem.
    max_index: jumlah indeks kamera yang akan diperiksa (0 sampai max_index-1)
    """
    available_cameras = []

    for index in range(max_index):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)  # CAP_DSHOW untuk Windows agar lebih cepat
        if cap.isOpened():
            print(f"Kamera ditemukan pada index {index}")
            available_cameras.append(index)
            cap.release()
        else:
            print(f"Tidak ada kamera pada index {index}")

    if not available_cameras:
        print("Tidak ada kamera yang terdeteksi.")
    else:
        print(f"Daftar kamera yang tersedia: {available_cameras}")

check_camera_indexes(5)  # cek index dari 0-4
