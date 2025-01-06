#include <Arduino_LSM9DS1.h>
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
//#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
//#include <tensorflow/lite/version.h>
#include <Wire.h>              // For I2C communication
#include <math.h>


// Include the model_tflite.h file 
#include "model_tflite.h" 

// Define tensor arena size (adjust depending on model's memory requirements)
constexpr int tensorArenaSize = 120000;  // Adjust this size based on your model's requirements
byte tensorArena[tensorArenaSize] __attribute__((aligned(16))); 

// Declare the TensorFlow Lite interpreter and resolver
const tflite::Model* model = nullptr;  // Declare model
tflite::AllOpsResolver resolver;  // Resolve operations used by the model
//flite::MicroErrorReporter errorReporter;
tflite::MicroInterpreter* interpreter = nullptr;  // Pointer to the interpreter

int ledR = D7;   // Red LED
int ledG = D4;   // Green LED
int piezo = D5;      // Piezo connected to digital pin 5

void setup() {
  Serial.begin(115200);
  
  // Initialise the IMU sensor (LSM9DS1)
  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU!");
    while (1);
  }

  // Initialise the TensorFlow Lite model
  model = tflite::GetModel(model_tflite);  // Load model from Flash memory
  if (model == nullptr) {
    Serial.println("Failed to load model!");
    while (1);
  }

  // Initialise the TensorFlow Lite interpreter
  interpreter = new tflite::MicroInterpreter(model, resolver, tensorArena, tensorArenaSize);
  interpreter->AllocateTensors();  // Allocate memory for tensors

  Serial.println("IMU and TensorFlow Lite model initialized.");

  // Set the LED and Piezo pins as outputs
  pinMode(ledR, OUTPUT);
  pinMode(ledG, OUTPUT);
  pinMode(piezo, OUTPUT);
}

// Set te same mean and std dev value from the training model
float Gyro_X_mean = 0.776791;     
float Gyro_X_std = 1.490773;      
float Gyro_Y_mean = -0.72206;
float Gyro_Y_std = 2.619044;
float Gyro_Z_mean = 1.575444;
float Gyro_Z_std = 1.432497;
float Accel_X_mean = 0.100546;
float Accel_X_std = 0.133388;
float Accel_Y_mean = 0.485571;
float Accel_Y_std = 0.363043;
float Accel_Z_mean = 0.66123;
float Accel_Z_std = 0.402161;

// Normalise function to apply the same formula as used during training
float normalize(float value, float mean, float std) {
  return (value - mean) / std;
}

void loop() {
  // Declare variables to hold the accelerometer and gyroscope data
  float Gyro_X, Gyro_Y, Gyro_Z, Accel_X, Accel_Y, Accel_Z;

  // Read gyroscope data
  if (IMU.gyroscopeAvailable()) {
    IMU.readGyroscope(Gyro_X, Gyro_Y, Gyro_Z);
  }

  // Read accelerometer data
  if (IMU.accelerationAvailable()) {
    IMU.readAcceleration(Accel_X, Accel_Y, Accel_Z);
  }

  // Normalize the sensor data
  Gyro_X = normalize(Gyro_X, Gyro_X_mean, Gyro_X_std);
  Gyro_Y = normalize(Gyro_Y, Gyro_Y_mean, Gyro_Y_std);
  Gyro_Z = normalize(Gyro_Z, Gyro_Z_mean, Gyro_Z_std);
  Accel_X = normalize(Accel_X, Accel_X_mean, Accel_X_std);
  Accel_Y = normalize(Accel_Y, Accel_Y_mean, Accel_Y_std);
  Accel_Z = normalize(Accel_Z, Accel_Z_mean, Accel_Z_std);

  // Calculate the magnitude of the gyroscope and accelerometer data
  float Gyro_magnitude = sqrt(Gyro_X * Gyro_X + Gyro_Y * Gyro_Y + Gyro_Z * Gyro_Z);
  float Accel_magnitude = sqrt(Accel_X * Accel_X + Accel_Y * Accel_Y + Accel_Z * Accel_Z);

  // Prepare input data for TensorFlow Lite model
  float input_data[8] = {Gyro_X, Gyro_Y, Gyro_Z, Gyro_magnitude, Accel_X, Accel_Y, Accel_Z, Accel_magnitude};
  
  // Set the input tensor with sensor data
  float* input = interpreter->input(0)->data.f;
  for (int i = 0; i < 8; i++) {
    input[i] = input_data[i];
  }

  // Run inference (make a prediction)
  interpreter->Invoke();

  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Failed to invoke model!");
    return;  // Exit if inference failed
  }


  // Get the output from the model
  float* output = interpreter->output(0)->data.f;

  //Serial.println(output[0]);

  // Output the prediction (1 = correct posture, 0 = wrong posture)
  
  if (output[1] > 0.5) {
    Serial.println("1");
    digitalWrite(ledG, HIGH);  // Turn on Green LED (Correct posture)
    digitalWrite(ledR, LOW);     // Turn off Red LED
    noTone(piezo);           // Turn off sound on Piezo

  } else {
    Serial.println("0");
    digitalWrite(ledG, LOW);  // Turn off Green LED (Correct posture)
    digitalWrite(ledR, HIGH);     // Turn on Red LED
    // Activate the piezo buzzer 3 times, 1 second each
    for (int i = 0; i < 3; i++) {
      tone(piezo, 500);         // Play sound on Piezo (500Hz)
      delay(1000);              // Wait for 1 second
      noTone(piezo);            // Stop sound on Piezo
      delay(500);               // Small delay between beeps
    }
  }
 
  // Delay before reading the next set of data
  delay(500);  // Adjust the delay time as needed
}
