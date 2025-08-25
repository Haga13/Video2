import serial

# Ganti 'COM3' jika port berbeda
PORT = 'COM3'
BAUDRATE = 115200

def main():
    try:
        ser = serial.Serial(PORT, BAUDRATE, timeout=2)
        print(f"Membuka port {PORT} pada baudrate {BAUDRATE}...")
        print("Tekan Ctrl+C untuk berhenti.")
        while True:
            line = ser.readline().decode(errors='ignore').strip()
            if line:
                print(f"Data diterima: {line}")
    except Exception as e:
        print(f"Gagal membuka serial: {e}")

if __name__ == "__main__":
    main()
