import processing.serial.*;

Serial myPort;
PrintWriter output;

void setup() {
  size(200, 200);
  myPort = new Serial(this, "COM7", 115200); // 修改为实际串口号
  output = createWriter("emg_data.csv");
}

void draw() {
  if (myPort.available() > 0) {
    String value = myPort.readStringUntil('\n');
    if (value != null) {
      output.println(value.trim());
    }
  }
}

void keyPressed() {
  output.flush();
  output.close();
  exit();
}