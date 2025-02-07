const float alpha = 0.2;  // 滤波系数（0.1-0.3）
int filteredValue = 0;

void setup() {
  Serial.begin(115200);
}

void loop() {
  int rawValue = analogRead(A0);

  // 1. 软件滤波
  filteredValue = alpha * rawValue + (1 - alpha) * filteredValue;

  // 2. 信号放大
  int amplifiedValue = map(filteredValue, 300, 700, 0, 1023);  // 动态放大有效范围
  amplifiedValue = constrain(amplifiedValue, 0, 1023);         // 限制输出范围

  // 输出结果
  Serial.println(amplifiedValue);
  delay(10);
}
