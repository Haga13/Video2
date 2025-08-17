/*
ESP32/ESP8266 Sensor Data Sender
Mengirim data sensor ke Python video streaming server
*/

#include <WiFi.h> // Untuk ESP32
// #include <ESP8266WiFi.h>  // Uncomment untuk ESP8266
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include <DHT.h>

// WiFi credentials
const char *ssid = "NAMA_WIFI_ANDA";
const char *password = "PASSWORD_WIFI_ANDA";

// Server settings
const char *serverURL = "http://192.168.1.100:5000/sensor_data"; // Ganti dengan IP server Python

// DHT sensor settings
#define DHT_PIN 2      // Pin DHT22
#define DHT_TYPE DHT22 // DHT 22 (AM2302)
DHT dht(DHT_PIN, DHT_TYPE);

// Timing
unsigned long lastSensorRead = 0;
const unsigned long sensorInterval = 5000; // Kirim data setiap 5 detik

void setup()
{
    Serial.begin(115200);

    // Initialize DHT sensor
    dht.begin();

    // Connect to WiFi
    WiFi.begin(ssid, password);
    Serial.print("Connecting to WiFi");

    while (WiFi.status() != WL_CONNECTED)
    {
        delay(500);
        Serial.print(".");
    }

    Serial.println();
    Serial.println("WiFi connected!");
    Serial.print("IP address: ");
    Serial.println(WiFi.localIP());
    Serial.print("Server URL: ");
    Serial.println(serverURL);
}

void loop()
{
    // Check WiFi connection
    if (WiFi.status() != WL_CONNECTED)
    {
        Serial.println("WiFi disconnected, reconnecting...");
        WiFi.begin(ssid, password);
        while (WiFi.status() != WL_CONNECTED)
        {
            delay(500);
            Serial.print(".");
        }
        Serial.println("WiFi reconnected!");
    }

    // Read and send sensor data
    if (millis() - lastSensorRead >= sensorInterval)
    {
        sendSensorData();
        lastSensorRead = millis();
    }

    delay(100);
}

void sendSensorData()
{
    // Read sensor data
    float temperature = dht.readTemperature();
    float humidity = dht.readHumidity();

    // Check if any reads failed
    if (isnan(humidity) || isnan(temperature))
    {
        Serial.println("Failed to read from DHT sensor!");
        return;
    }

    // Print sensor data
    Serial.println("=== SENSOR DATA ===");
    Serial.print("Temperature: ");
    Serial.print(temperature);
    Serial.println("°C");
    Serial.print("Humidity: ");
    Serial.print(humidity);
    Serial.println("%");

    // Create JSON payload
    StaticJsonDocument<200> doc;
    doc["temperature"] = temperature;
    doc["humidity"] = humidity;
    doc["timestamp"] = millis();
    doc["device_id"] = "ESP32_SENSOR_01";

    String jsonString;
    serializeJson(doc, jsonString);

    // Send HTTP POST request
    if (WiFi.status() == WL_CONNECTED)
    {
        HTTPClient http;
        http.begin(serverURL);
        http.addHeader("Content-Type", "application/json");

        int httpResponseCode = http.POST(jsonString);

        if (httpResponseCode > 0)
        {
            String response = http.getString();
            Serial.print("HTTP Response: ");
            Serial.println(httpResponseCode);
            Serial.print("Response: ");
            Serial.println(response);

            if (httpResponseCode == 200)
            {
                Serial.println("✅ Data sent successfully!");
            }
        }
        else
        {
            Serial.print("❌ Error sending data: ");
            Serial.println(httpResponseCode);
        }

        http.end();
    }
    else
    {
        Serial.println("❌ WiFi not connected");
    }

    Serial.println("==================");
}

// Fungsi untuk mengirim data sensor tambahan (opsional)
void sendCustomSensorData(float value1, float value2, String sensorType)
{
    StaticJsonDocument<200> doc;
    doc["temperature"] = value1;
    doc["humidity"] = value2;
    doc["sensor_type"] = sensorType;
    doc["timestamp"] = millis();
    doc["device_id"] = "ESP32_CUSTOM";

    String jsonString;
    serializeJson(doc, jsonString);

    HTTPClient http;
    http.begin(serverURL);
    http.addHeader("Content-Type", "application/json");

    int httpResponseCode = http.POST(jsonString);

    if (httpResponseCode == 200)
    {
        Serial.println("✅ Custom data sent!");
    }
    else
    {
        Serial.println("❌ Failed to send custom data");
    }

    http.end();
}
