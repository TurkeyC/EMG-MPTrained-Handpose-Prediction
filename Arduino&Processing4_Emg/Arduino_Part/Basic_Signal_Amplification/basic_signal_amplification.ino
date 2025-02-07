void setup() {
  Serial.begin(115200);
}

void loop() {
  int rawValue = analogRead(A0);
  int amplifiedValue = map(rawValue, 300, 700, 0, 1023);  // 动态放大有效范围
  amplifiedValue = constrain(amplifiedValue, 0, 1023);    // 限制输出范围
  Serial.println(amplifiedValue);
  delay(10);
}
