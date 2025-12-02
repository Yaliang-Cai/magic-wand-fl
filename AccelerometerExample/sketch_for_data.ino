/*
  IMU Capture
  This sketch captures accelerometer and gyroscope data to train a TinyML model.
  It is designed to work with the "Magic Wand Gesture Recorder" web page.
*/

#include <ArduinoBLE.h>
#include <Arduino_LSM9DS1.h> 

// BLE Service and Characteristic UUIDs (Must match the HTML page)
#define SERVICE_UUID        "4798e0f2-0000-4d68-af64-8a8f5258404e"
#define CHARACTERISTIC_UUID "4798e0f2-300a-4d68-af64-8a8f5258404e"

// Configuration
const int maxSamples = 160; // Max samples per gesture
const float accelerationThreshold = 1.2; // Threshold to start recording (in Gs)
const int gyroscopeThreshold = 10; // Threshold to start recording (in deg/s)

BLEService service(SERVICE_UUID);
BLECharacteristic strokeCharacteristic(CHARACTERISTIC_UUID, BLERead | BLENotify, 2048);

// Buffer to hold the gesture data
// Format: [state, count, x1, y1, x2, y2, ...]
// state: 0=idle, 1=drawing, 2=finished
uint8_t strokeData[2048]; 
int sampleCount = 0;
bool recording = false;

void setup() {
  Serial.begin(9600);
  
  // Initialize IMU
  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU!");
    while (1);
  }
  Serial.println("IMU initialized.");

  // Initialize BLE
  if (!BLE.begin()) {
    Serial.println("starting BLE failed!");
    while (1);
  }

  BLE.setLocalName("BLESense");
  BLE.setAdvertisedService(service);
  service.addCharacteristic(strokeCharacteristic);
  BLE.addService(service);
  BLE.advertise();

  Serial.println("Bluetooth device active, waiting for connections...");
}

void loop() {
  BLEDevice central = BLE.central();

  if (central) {
    Serial.print("Connected to central: ");
    Serial.println(central.address());

    while (central.connected()) {
      float aX, aY, aZ, gX, gY, gZ;

      if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) {
        IMU.readAcceleration(aX, aY, aZ);
        IMU.readGyroscope(gX, gY, gZ);

        // Calculate magnitude to detect motion
        float aSum = fabs(aX) + fabs(aY) + fabs(aZ);
        float gSum = fabs(gX) + fabs(gY) + fabs(gZ);

        if (!recording) {
          // Check if we should start recording
          if (aSum > accelerationThreshold || gSum > gyroscopeThreshold) {
            recording = true;
            sampleCount = 0;
            Serial.println("Gesture started");
            
            // Set state to 1 (Drawing)
             int32_t state = 1;
             int32_t count = 0;
             memcpy(strokeData, &state, 4);
             memcpy(strokeData + 4, &count, 4);
             strokeCharacteristic.writeValue(strokeData, 8);
          }
        } 
        
        if (recording) {
          // Record data
          // Map sensor data to byte range for efficiency (simple normalization)
          // Note: This mapping needs to align with what the JS expects. 
          // The standard sketch usually maps -2.0G..2.0G to -128..127
          
          int8_t x_val = constrain((aX) * 64.0, -128, 127);
          int8_t y_val = constrain((aY) * 64.0, -128, 127);
          // Only using X and Y accel for the 2D canvas drawing in standard demo

          int index = 8 + (sampleCount * 2); 
          strokeData[index] = (uint8_t)x_val;
          strokeData[index + 1] = (uint8_t)y_val;
          
          sampleCount++;

          // Check if gesture is finished (too long or stopped moving)
          // Simple stop condition: Motion drops below threshold? 
          // Or just fill the buffer. The standard sketch often fills buffer or timeouts.
          // Here we just fill the buffer to maxSamples for simplicity or wait for stop.
          
          if (sampleCount >= maxSamples || (aSum < 1.1 && gSum < 5.0 && sampleCount > 10)) {
             recording = false;
             Serial.println("Gesture finished");

             // Update header
             int32_t state = 2; // Finished
             int32_t count = sampleCount;
             memcpy(strokeData, &state, 4);
             memcpy(strokeData + 4, &count, 4);
             
             // Send full packet
             strokeCharacteristic.writeValue(strokeData, 328);
          }
        }
      }
    }
    Serial.println("Disconnected");
  }
}