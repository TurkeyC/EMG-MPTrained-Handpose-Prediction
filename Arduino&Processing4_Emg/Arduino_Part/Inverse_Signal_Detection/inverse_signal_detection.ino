void setup() {
  Serial.begin(115200);
}

void loop() {
  int rawValue = analogRead(A0);
  int invertedValue = 1023 - rawValue;  // 反向信号
  Serial.print(rawValue);
  Serial.print(",");
  Serial.println(invertedValue);  // 同时输出原始值和反向值
  delay(10);
}