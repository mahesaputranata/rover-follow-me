// ================================================================
//  Rover ESP32 Firmware — v2 (PID velocity control)
//  Builds on v1 serial protocol. Changes are marked with [NEW].
//
//  Protocol (Pi → ESP32):
//    SET_VEL <left_mps> <right_mps>   [NEW] target velocity m/s
//    STOP                              stops motors, disables PID
//    PING                              → PONG
//    GET_STATUS                        → STATUS ...
//    RESET_ENCODERS                    → OK RESET_ENCODERS
//    AUTO FORWARD_TEST                 autonomous test sequence
//    AUTO OFF
//
//  Protocol (ESP32 → Pi), streamed every TELEM_INTERVAL_MS:
//    ODOM <dA> <dB> <dt_ms>           [NEW] delta ticks + dt
//    STATUS L=.. R=.. ...             unchanged
// ================================================================

// -------- Motor A pins --------
const int DIR_A = 26;
const int PWM_A = 25;

// -------- Motor B pins --------
const int DIR_B = 27;
const int PWM_B = 14;


// -------- Encoder A pins --------
const int ENC_A1 = 16; //purple
const int ENC_A2 = 17; //blue

// -------- Encoder B pins --------
const int ENC_B1 = 18; // black -> debug change to D18
const int ENC_B2 = 19; // white -> debug change to D19

// -------- PWM settings --------
const int PWM_FREQ       = 20000;
const int PWM_RESOLUTION = 8;      // 0–255


const float WHEEL_DIAMETER_M    = 0.065f;   // 65 mm wheels — adjust
const int   TICKS_PER_REV       = 1440;     // encoder ticks per wheel revolution — adjust (should be based on measured, 1440 still not based on meaasurement)
const float WHEEL_CIRCUMFERENCE = 3.14159f * WHEEL_DIAMETER_M;
const float METERS_PER_TICK     = WHEEL_CIRCUMFERENCE / (float)TICKS_PER_REV;
const float KP = 200.0f;          // was 80
const float KI = 400.0f;          // was 0

const float FF_PWM = 100.0f;      // existing static-friction feedforward
const float I_PWM_LIMIT = 120.0f; // max PWM the integral may contribute
const float SP_DEADBAND = 0.08f;  // |setpoint| below this = stop (no ±100 flip)
const int   PID_PRINT_EVERY = 10; // print 1 PID line per 10 ticks (5 Hz)

const float KD =  0.0f;

// PID loop interval
const int PID_INTERVAL_MS   = 20;    // 50 Hz control loop
const int TELEM_INTERVAL_MS = 20;    // 50 Hz odometry stream to Pi

// Watchdog: stop motors if no SET_VEL received within this window
const unsigned long WATCHDOG_MS = 2000;



struct PIDState {
  float setpoint;     // target velocity m/s
  float integral;
  float prevError;
  int   output;       // PWM value (-255 .. 255)
};

PIDState pidA = {0, 0, 0, 0};
PIDState pidB = {0, 0, 0, 0};

bool pidEnabled = false;

// ================================================================
//  Timing
// ================================================================
unsigned long lastPidTime      = 0;
unsigned long lastTelemTime    = 0;
unsigned long lastCmdTime      = 0;  // watchdog

// ================================================================
//  Auto mode (unchanged from v1)
// ================================================================
bool     autoMode       = false;
String   autoModeName   = "OFF";
unsigned long autoLastStepTime = 0;
int      autoStage      = 0;

// Serial input buffer
String inputBuffer = "";

// ================================================================
// Encoder state
// ================================================================
volatile int32_t encoderCountA = 0;
volatile int32_t encoderCountB = 0;

volatile uint8_t previousStateA = 0;
volatile uint8_t previousStateB = 0;

// Keep lookup data in RAM because it is accessed inside an ISR.
DRAM_ATTR const int8_t QUADRATURE_TABLE[16] = {
   0, -1,  1,  0,
   1,  0,  0, -1,
  -1,  0,  0,  1,
   0,  1, -1,  0
};

// Snapshot taken at the start of each PID tick
int32_t prevTickA = 0;
int32_t prevTickB = 0;

// ================================================================
//  Encoder ISRs — unchanged from v1
// ================================================================
void IRAM_ATTR handleEncoderA() {
  uint8_t currentState =
      (digitalRead(ENC_A1) << 1) |
       digitalRead(ENC_A2);

  uint8_t transition =
      (previousStateA << 2) | currentState;

  encoderCountA += QUADRATURE_TABLE[transition];
  previousStateA = currentState;
}

void IRAM_ATTR handleEncoderB() {
  uint8_t currentState =
      (digitalRead(ENC_B1) << 1) |
       digitalRead(ENC_B2);

  uint8_t transition =
      (previousStateB << 2) | currentState;

  encoderCountB -= QUADRATURE_TABLE[transition];
  previousStateB = currentState;
}

// ================================================================
//  Motor helpers — unchanged from v1
// ================================================================
int clampPWM(int v) {
  if (v >  255) return  255;
  if (v < -255) return -255;
  return v;
}

void applyMotorA(int pwm) {
  pwm = clampPWM(pwm);
  digitalWrite(DIR_A, pwm >= 0 ? LOW : HIGH);   // was HIGH : LOW
  ledcWrite(PWM_A, abs(pwm));
}

void applyMotorB(int pwm) {
  pwm = clampPWM(pwm);
  digitalWrite(DIR_B, pwm >= 0 ? HIGH : LOW);   // was LOW : HIGH
  ledcWrite(PWM_B, abs(pwm));
}

void stopAllMotors() {
  ledcWrite(PWM_A, 0);
  ledcWrite(PWM_B, 0);
  pidA = {0, 0, 0, 0};
  pidB = {0, 0, 0, 0};
  pidEnabled = false;
}

int pidUpdate(PIDState &p, float vel, float dt) {
  float sp = (fabsf(p.setpoint) < SP_DEADBAND) ? 0.0f : p.setpoint;
  if (sp == 0.0f) {                       // stopped: gentle brake, no windup
    p.integral = 0.0f;
    p.prevError = 0.0f;
    if (fabsf(vel) < 0.02f) return 0;
    return clampPWM((int)(KP * (0.0f - vel)));
  }
  float error = sp - vel;
  p.integral += error * dt;
  float lim = I_PWM_LIMIT / KI;           // clamp in PWM terms
  p.integral = constrain(p.integral, -lim, lim);
  p.prevError = error;
  float ff = (sp > 0) ? FF_PWM : -FF_PWM;
  return clampPWM((int)(ff + KP * error + KI * p.integral));
}

void runPID(unsigned long nowMs) {
  int32_t dtMs = (int32_t)(nowMs - lastPidTime);  // signed: first tick after SET_VEL can be negative
  if (dtMs <= 0) return;
  float dt = dtMs / 1000.0f;

  noInterrupts();
  long curA = encoderCountA;
  long curB = encoderCountB;
  interrupts();

  long deltaA = curA - prevTickA;
  long deltaB = curB - prevTickB;
  prevTickA = curA;
  prevTickB = curB;

  float velA = (deltaA * METERS_PER_TICK) / dt;
  float velB = (deltaB * METERS_PER_TICK) / dt;

  pidA.output = pidUpdate(pidA, velA, dt);
  pidB.output = pidUpdate(pidB, velB, dt);
  applyMotorA(pidA.output);
  applyMotorB(pidB.output);

  static uint8_t printCount = 0;
  if (++printCount >= PID_PRINT_EVERY) {
    printCount = 0;
    Serial.print("PID dt="); Serial.print(dt, 4);
    Serial.print(" velA="); Serial.print(velA, 4);
    Serial.print(" velB="); Serial.print(velB, 4);
    Serial.print(" spA="); Serial.print(pidA.setpoint, 4);
    Serial.print(" spB="); Serial.print(pidB.setpoint, 4);
    Serial.print(" pwmA="); Serial.print(pidA.output);
    Serial.print(" pwmB="); Serial.println(pidB.output);
  }
}

void streamOdom(unsigned long nowMs) {
  static int32_t lastA = 0;
  static int32_t lastB = 0;
  static unsigned long lastTime = 0;

  if (lastTime == 0) {
    noInterrupts();
    lastA = encoderCountA;
    lastB = encoderCountB;
    interrupts();

    lastTime = nowMs;
    return;
  }

  noInterrupts();
  int32_t curA = encoderCountA;
  int32_t curB = encoderCountB;
  interrupts();

  int32_t dA = curA - lastA;
  int32_t dB = curB - lastB;
  unsigned long dt = nowMs - lastTime;

  lastA = curA;
  lastB = curB;
  lastTime = nowMs;

  Serial.print("ODOM ");
  Serial.print(dA);
  Serial.print(" ");
  Serial.print(dB);
  Serial.print(" ");
  Serial.println(dt);
}


void printStatus() {
  Serial.print("STATUS ");
  Serial.print("L="); Serial.print(pidA.setpoint);
  Serial.print(" R="); Serial.print(pidB.setpoint);
  Serial.print(" PWM_A="); Serial.print(pidA.output);
  Serial.print(" PWM_B="); Serial.print(pidB.output);
  Serial.print(" AUTO="); Serial.print(autoMode ? autoModeName : "OFF");
  Serial.print(" ENC_A="); Serial.print(encoderCountA);
  Serial.print(" ENC_B="); Serial.println(encoderCountB);
}

void resetEncoders() {
  noInterrupts();
  encoderCountA = 0;
  encoderCountB = 0;
  interrupts();
  prevTickA = 0;
  prevTickB = 0;
}

// ================================================================
//  Auto mode — updated to use SET_VEL style (m/s) internally
// ================================================================
void runAutoForwardTest() {
  unsigned long now = millis();
  if (now - autoLastStepTime < 2000) return;
  autoLastStepTime = now;

  switch (autoStage) {
    case 0:
      pidA.setpoint = 0.2f;  pidB.setpoint = 0.2f;
      pidEnabled = true;
      Serial.println("OK AUTO_STAGE FORWARD_0.2mps");
      break;
    case 1:
      stopAllMotors();
      Serial.println("OK AUTO_STAGE STOP");
      break;
    case 2:
      pidA.setpoint = -0.2f; pidB.setpoint = -0.2f;
      pidEnabled = true;
      Serial.println("OK AUTO_STAGE REVERSE_0.2mps");
      break;
    case 3:
      stopAllMotors();
      Serial.println("OK AUTO_STAGE STOP");
      break;
    case 4:
      pidA.setpoint = 0.2f;  pidB.setpoint = 0.0f;
      pidEnabled = true;
      Serial.println("OK AUTO_STAGE LEFT_ONLY");
      break;
    case 5:
      stopAllMotors();
      Serial.println("OK AUTO_STAGE STOP");
      break;
    case 6:
      pidA.setpoint = 0.0f;  pidB.setpoint = 0.2f;
      pidEnabled = true;
      Serial.println("OK AUTO_STAGE RIGHT_ONLY");
      break;
    default:
      stopAllMotors();
      autoStage = -1;
      Serial.println("OK AUTO_STAGE DONE");
      break;
  }
  autoStage++;
}


// ================================================================
//  Command parser
// ================================================================
void processCommand(String cmd) {
  cmd.trim();

  if (cmd.length() == 0) return;

  // --- PING ---
  if (cmd == "PING") {
    Serial.println("PONG");
    return;
  }

  // --- STOP ---
  if (cmd == "STOP") {
    autoMode = false;
    autoModeName = "OFF";
    stopAllMotors();
    Serial.println("OK STOP");
    return;
  }


  if (cmd.startsWith("SET_VEL ")) {
      String args     = cmd.substring(8);
      int spaceIdx    = args.indexOf(' ');
      if (spaceIdx > 0) {
          float leftMps  = args.substring(0, spaceIdx).toFloat();
          float rightMps = args.substring(spaceIdx + 1).toFloat();

          autoMode      = false;
          autoModeName  = "OFF";

          if (!pidEnabled) {
              noInterrupts();
              prevTickA = encoderCountA;
              prevTickB = encoderCountB;
              interrupts();

              lastPidTime = millis();

              pidA.integral = 0.0f;
              pidA.prevError = 0.0f;
              pidA.output = 0;

              pidB.integral = 0.0f;
              pidB.prevError = 0.0f;
              pidB.output = 0;
          }

          pidA.setpoint = leftMps;
          pidB.setpoint = rightMps;
          pidEnabled    = true;
          lastCmdTime   = millis();

          Serial.print("OK SET_VEL ");
          Serial.print(leftMps);
          Serial.print(" ");
          Serial.println(rightMps);
          return;
      }
  }

  // --- RESET_ENCODERS ---
  if (cmd == "RESET_ENCODERS") {
    resetEncoders();
    Serial.println("OK RESET_ENCODERS");
    return;
  }

  // --- GET_STATUS ---
  if (cmd == "GET_STATUS") {
    printStatus();
    return;
  }

  // --- MOVE (raw PWM, kept for manual bench testing only) ---
  int leftPWM, rightPWM;
  if (sscanf(cmd.c_str(), "MOVE %d %d", &leftPWM, &rightPWM) == 2) {
    autoMode     = false;
    pidEnabled   = false;
    pidA.setpoint = 0; pidB.setpoint = 0;
    applyMotorA(leftPWM);
    applyMotorB(rightPWM);
    Serial.print("OK MOVE ");
    Serial.print(leftPWM);
    Serial.print(" ");
    Serial.println(rightPWM);
    return;
  }

  // --- AUTO ---
  char autoArg[32];
  if (sscanf(cmd.c_str(), "AUTO %31s", autoArg) == 1) {
    String mode = String(autoArg);
    if (mode == "FORWARD_TEST") {
      autoMode         = true;
      autoModeName     = mode;
      autoStage        = 0;
      autoLastStepTime = 0;
      Serial.println("OK AUTO FORWARD_TEST");
      return;
    }
    if (mode == "OFF") {
      autoMode     = false;
      autoModeName = "OFF";
      stopAllMotors();
      Serial.println("OK AUTO OFF");
      return;
    }
    Serial.println("ERR UNKNOWN_AUTO_MODE");
    return;
  }

  Serial.println("ERR UNKNOWN_COMMAND");
}

// ================================================================
//  Setup
// ================================================================
void setup() {
  Serial.begin(115200);
  delay(500);

  pinMode(DIR_A, OUTPUT);
  pinMode(DIR_B, OUTPUT);

  ledcAttach(PWM_A, PWM_FREQ, PWM_RESOLUTION);
  ledcAttach(PWM_B, PWM_FREQ, PWM_RESOLUTION);

  pinMode(ENC_A1, INPUT_PULLUP);
  pinMode(ENC_A2, INPUT_PULLUP);
  pinMode(ENC_B1, INPUT_PULLUP);
  pinMode(ENC_B2, INPUT_PULLUP);

  previousStateA =
      (digitalRead(ENC_A1) << 1) |
      digitalRead(ENC_A2);

  previousStateB =
      (digitalRead(ENC_B1) << 1) |
      digitalRead(ENC_B2);

  attachInterrupt(
      digitalPinToInterrupt(ENC_A1),
      handleEncoderA,
      CHANGE
  );

  attachInterrupt(
      digitalPinToInterrupt(ENC_A2),
      handleEncoderA,
      CHANGE
  );

  attachInterrupt(
      digitalPinToInterrupt(ENC_B1),
      handleEncoderB,
      CHANGE
  );

  attachInterrupt(
      digitalPinToInterrupt(ENC_B2),
      handleEncoderB,
      CHANGE
  );

  stopAllMotors();
  lastCmdTime = millis();
  inputBuffer.reserve(64);

  Serial.println("READY v2");
}

// ================================================================
//  Main loop
// ================================================================

void loop() {
  unsigned long now = millis();

  // --- read serial commands ---
  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\n') {
      processCommand(inputBuffer);
      inputBuffer = "";
    } else if (c != '\r') {
      inputBuffer += c;
    }
  }

  // --- [NEW] PID control loop at 50 Hz ---
  if (pidEnabled && (now - lastPidTime >= PID_INTERVAL_MS)) {
    runPID(now);
    lastPidTime = now;
  }

  // --- [NEW] watchdog: kill motors if Pi goes silent ---
  if (pidEnabled && (millis() - lastCmdTime > WATCHDOG_MS)) {
    stopAllMotors();
    Serial.println("WARN WATCHDOG_STOP");
  }

  // --- [NEW] odometry telemetry at 50 Hz ---
  if (now - lastTelemTime >= TELEM_INTERVAL_MS) {
    streamOdom(now);
    lastTelemTime = now;
  }

  // --- auto mode ---
  if (autoMode && autoModeName == "FORWARD_TEST") {
    runAutoForwardTest();
  }
}
