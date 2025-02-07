// 肌电信号采集与滤波程序
// 版本：1.0
// 功能：实时采集肌电信号并进行移动平均滤波

// ================== 参数设置 ==================
const int sensorPin = A0;       // 肌电信号输入引脚
const int numReadings = 20;     // 移动平均窗口大小（建议10-20）
const int sampleInterval = 20;  // 采样间隔（毫秒）

// ================== 全局变量 ==================
int readings[numReadings];      // 存储历史读数
int readIndex = 0;              // 当前读数索引
int total = 0;                  // 读数总和
int average = 0;                // 移动平均值

// ================== 初始化 ==================
void setup() {
  // 初始化串口通信
  Serial.begin(115200);

  // 初始化读数数组
  for (int i = 0; i < numReadings; i++) {
    readings[i] = 0;
  }
}

// ================== 主循环 ==================
void loop() {
  // 1. 读取原始信号
  int rawValue = analogRead(sensorPin);

  // 2. 更新移动平均
  total -= readings[readIndex];          // 减去最旧的读数
  readings[readIndex] = rawValue;        // 存储新读数
  total += readings[readIndex];          // 加上最新读数
  readIndex = (readIndex + 1) % numReadings; // 更新索引

  // 3. 计算移动平均值
  average = total / numReadings;

  // 4. 输出结果
  Serial.print("Raw:");
  Serial.print(rawValue);
  Serial.print(" Filtered:");
  Serial.println(average);

  // 5. 等待下次采样
  delay(sampleInterval);
}