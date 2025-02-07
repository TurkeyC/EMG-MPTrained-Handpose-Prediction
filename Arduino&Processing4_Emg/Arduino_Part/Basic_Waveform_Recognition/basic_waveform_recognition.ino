void setup() {
  Serial.begin(115200); // 设置串口波特率
}

void loop() {
  int rawValue = analogRead(A0);  // 读取A0引脚数据
  Serial.println(rawValue);       // 输出原始值
  delay(20);                      // 采样间隔约100Hz
}