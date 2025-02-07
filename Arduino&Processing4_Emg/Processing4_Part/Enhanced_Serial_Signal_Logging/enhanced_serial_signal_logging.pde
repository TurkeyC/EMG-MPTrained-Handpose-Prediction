import processing.serial.*;

Serial myPort;        // 串口对象
PrintWriter output;   // 文件输出对象
boolean recording = false; // 记录状态

void setup() {
  size(300, 200);  // 窗口大小
  println(Serial.list()); // 打印可用串口
  myPort = new Serial(this, "COM7", 115200); // 修改为实际串口号
  textSize(32);
}

void draw() {
  background(255);
  if (recording) {
    fill(255, 0, 0);
    text("Recording...", 20, 100);
  } else {
    fill(0);
    text("Press SPACE to start", 20, 100);
  }

  // 读取串口数据
  while (myPort.available() > 0) {
    String value = myPort.readStringUntil('\n');
    if (value != null && recording) {
      value = value.trim(); // 去除空白字符
      output.println(value); // 写入文件
    }
  }
}

void keyPressed() {
  if (key == ' ') {  // 空格键控制
    if (!recording) {
      output = createWriter("emg_data.csv"); // 创建文件
      recording = true;
    } else {
      output.flush(); // 保存数据
      output.close();
      recording = false;
      println("Data saved!");
    }
  }
}