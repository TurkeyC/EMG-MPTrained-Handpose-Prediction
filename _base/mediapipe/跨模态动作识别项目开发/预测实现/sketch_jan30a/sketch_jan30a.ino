void setup() {
  Serial.begin(115200);  // 设置波特率
  // Arduino Uno的analogRead默认就是10位精度，无需额外设置
}

void loop() {
  int rawValue = analogRead(A0);  // 读取A0引脚（10位精度，0-1023）
  Serial.println(rawValue);       // 发送原始值
  delayMicroseconds(900);         // 控制采样率约1000Hz
}
