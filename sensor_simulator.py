#!/usr/bin/env python3
"""
Sensor Data Simulator
Script untuk simulasi data sensor ESP sebelum ESP asli ready
"""

import requests
import time
import random
import json

# Server configuration
SERVER_URL = "http://localhost:5000/sensor_data"

def send_sensor_data(temperature, humidity):
    """Kirim data sensor ke server"""
    data = {
        "temperature": temperature,
        "humidity": humidity,
        "timestamp": time.time(),
        "device_id": "SIMULATOR_01"
    }
    
    try:
        response = requests.post(SERVER_URL, json=data, timeout=5)
        if response.status_code == 200:
            print(f"✅ Data sent: Temp={temperature:.1f}°C, Humidity={humidity:.1f}%")
            return True
        else:
            print(f"❌ Error: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error sending data: {e}")
        return False

def simulate_realistic_data():
    """Simulasi data sensor yang realistis"""
    base_temp = 25.0  # Suhu dasar
    base_humidity = 60.0  # Kelembaban dasar
    
    while True:
        # Variasi suhu: 20-35°C
        temperature = base_temp + random.uniform(-5, 10) + (random.random() - 0.5) * 2
        
        # Variasi kelembaban: 40-80%
        humidity = base_humidity + random.uniform(-20, 20) + (random.random() - 0.5) * 5
        
        # Batasi range
        temperature = max(15, min(40, temperature))
        humidity = max(30, min(90, humidity))
        
        # Kirim data
        send_sensor_data(temperature, humidity)
        
        # Tunggu 3 detik
        time.sleep(3)

def simulate_extreme_conditions():
    """Simulasi kondisi ekstrem untuk testing"""
    conditions = [
        (45.0, 90.0),  # Panas + Lembab
        (15.0, 30.0),  # Dingin + Kering
        (35.0, 25.0),  # Panas + Kering
        (20.0, 85.0),  # Dingin + Lembab
    ]
    
    for temp, humidity in conditions:
        print(f"🔥 Testing extreme condition: {temp}°C, {humidity}%")
        send_sensor_data(temp, humidity)
        time.sleep(2)

def manual_input():
    """Input manual untuk testing"""
    print("=== Manual Sensor Data Input ===")
    print("Masukkan 'q' untuk keluar")
    
    while True:
        try:
            temp_input = input("Temperature (°C): ")
            if temp_input.lower() == 'q':
                break
                
            humidity_input = input("Humidity (%): ")
            if humidity_input.lower() == 'q':
                break
            
            temperature = float(temp_input)
            humidity = float(humidity_input)
            
            send_sensor_data(temperature, humidity)
            
        except ValueError:
            print("❌ Input tidak valid! Masukkan angka.")
        except KeyboardInterrupt:
            break
    
    print("👋 Manual input selesai")

def main():
    print("🌡️ SENSOR DATA SIMULATOR")
    print("=" * 40)
    print("1. Simulasi data realistis (otomatis)")
    print("2. Test kondisi ekstrem")
    print("3. Input manual")
    print("4. Keluar")
    
    choice = input("\nPilih mode (1-4): ").strip()
    
    if choice == "1":
        print("🤖 Memulai simulasi data realistis...")
        print("Tekan Ctrl+C untuk berhenti")
        try:
            simulate_realistic_data()
        except KeyboardInterrupt:
            print("\n⏹️ Simulasi dihentikan")
    
    elif choice == "2":
        print("🔥 Testing kondisi ekstrem...")
        simulate_extreme_conditions()
        print("✅ Test selesai")
    
    elif choice == "3":
        manual_input()
    
    elif choice == "4":
        print("👋 Bye!")
    
    else:
        print("❌ Pilihan tidak valid")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Program dihentikan")
