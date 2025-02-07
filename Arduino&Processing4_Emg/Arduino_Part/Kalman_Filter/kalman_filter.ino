float Q = 0.01;  // 过程噪声协方差
float R = 0.1;   // 观测噪声协方差
float P = 1.0;   // 估计误差协方差
float K = 0;     // 卡尔曼增益
float X = 0;     // 估计值

void setup() {
  Serial.begin(115200);
}

void loop() {
  int rawValue = analogRead(A0);
  float filteredValue = kalmanFilter(rawValue);

//  Serial.print("Raw:");
//  Serial.print(rawValue);
  Serial.print(" Kalman:");
  Serial.println(filteredValue);

  delay(20);
}

float kalmanFilter(float measurement) {
  // 预测
  P = P + Q;

  // 更新
  K = P / (P + R);
  X = X + K * (measurement - X);
  P = (1 - K) * P;

  return X;
}