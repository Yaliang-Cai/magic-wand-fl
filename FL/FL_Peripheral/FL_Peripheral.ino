#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <TinyMLShield.h>
#include <ArduinoBLE.h>

#define LEARNING_RATE 0.005
#define DATA_TYPE_FLOAT
#define DEBUG 0

#include "FL_Config.h"

extern const int first_layer_input_cnt;
extern const int classes_cnt;

static const unsigned int NN_def[] = {first_layer_input_cnt, 20, classes_cnt};

#include "Client_B_data.h"      // <--- B 节点数据 (训练缺1)
#include "NN_functions.h"

// BLE 服务定义
BLEService flService(FL_SERVICE_UUID);
BLECharacteristic weightChar(FL_WEIGHT_CHAR_UUID, BLERead | BLEWrite, CHUNK_SIZE_FLOATS * sizeof(float)); 
BLEByteCharacteristic statusChar(FL_STATUS_CHAR_UUID, BLERead | BLEWrite);

int iter_cnt = 0;
int current_round = 0;
int weights_count = 0;
// DATA_TYPE* WeightBiasPtr = NULL; // ⚠️ 已在 NN_functions.h 中定义，这里必须注释掉

// 🔥 新增：运行状态标志位
bool fl_running = false;

void setup() {
  Serial.begin(9600);
  delay(5000);
  
  initializeShield();
  srand(0);

  if (!BLE.begin()) {
    Serial.println("BLE failed!");
    while (1);
  }

  // 初始化神经网络
  weights_count = calcTotalWeightsBias();
  WeightBiasPtr = (DATA_TYPE*) calloc(weights_count, sizeof(DATA_TYPE));
  setupNN(WeightBiasPtr);

  // 设置 BLE
  BLE.setLocalName("FL_Node_B");
  BLE.setAdvertisedService(flService);
  flService.addCharacteristic(weightChar);
  flService.addCharacteristic(statusChar);
  BLE.addService(flService);
  
  statusChar.writeValue(STATUS_TRAINING);
  BLE.advertise();

  Serial.println("Arduino B (Peripheral) Ready.");
  Serial.println("Press button ONCE to start FULL AUTOMATED FL.");
}

// 执行本地训练和通信等待
void do_round_task() {
  Serial.print("\n=== ROUND "); Serial.print(current_round + 1); Serial.println(" ===");

  // 1. 本地训练
  Serial.println("[Local] Training started...");
  shuffleIndx();
  for (int i = 0; i < EPOCHS_PER_ROUND; i++) {
    for (int j = 0; j < train_data_cnt; j++) {
       generateTrainVectors(j); 
       forwardProp(); 
       backwardProp(); 
    }
    iter_cnt++;
  }
  Serial.println("[Local] Training Done.");
  // printAccuracy(); // 自动模式下这一行可以注释掉，减少刷屏，以 Central 的输出为主

  // 2. 准备数据
  packUnpackVector(0); // PACK
  
  // 3. 进入等待模式 (BLE)
  Serial.println("[BLE] Advertising... Waiting for Server connection...");
  statusChar.writeValue(STATUS_READY_TO_AGGREGATE);

  // 循环等待直到聚合完成
  bool done = false;
  unsigned long waitStart = millis(); // 可选：加入超时检测
  
  while (!done) {
    BLE.poll();
    byte status = statusChar.value();

    // 状态机处理
    if (status == STATUS_A_READ_ACK) {
       // A 说它准备好读了，我们开始发
       int sent_count = 0;
       while (sent_count < weights_count) {
          int chunk_len = CHUNK_SIZE_FLOATS;
          if (sent_count + chunk_len > weights_count) chunk_len = weights_count - sent_count;
          
          weightChar.writeValue((uint8_t*)(WeightBiasPtr + sent_count), chunk_len * sizeof(float));
          statusChar.writeValue(STATUS_B_SENDING_CHUNK);
          
          // 等待 A 读完
          while (statusChar.value() != STATUS_A_READ_ACK) { BLE.poll(); }
          sent_count += chunk_len;
       }
       statusChar.writeValue(STATUS_B_SENDING_COMPLETE);
    }
    else if (status == STATUS_A_SENDING_CHUNK) {
       // A 发回了新权重，我们要读
       static int recv_count = 0; 
       int chunk_len = CHUNK_SIZE_FLOATS;
       if (recv_count + chunk_len > weights_count) chunk_len = weights_count - recv_count;
       
       weightChar.readValue((uint8_t*)(WeightBiasPtr + recv_count), chunk_len * sizeof(float));
       recv_count += chunk_len;
       
       statusChar.writeValue(STATUS_B_READ_ACK);
       
       if (recv_count >= weights_count) recv_count = 0; // 重置
    }
    else if (status == STATUS_AGGREGATION_DONE) {
       done = true;
    }
  }

  // 4. 更新模型
  packUnpackVector(1); // UNPACK
  Serial.println(">>> ROUND COMPLETE: Model Updated <<<\n");
  
  current_round++;
  statusChar.writeValue(STATUS_TRAINING);
}

void loop() {
  // 1. 等待启动信号 (只按一次)
  if (!fl_running) {
    if (readShieldButton()) {
      fl_running = true;
      Serial.println("\n>>> AUTOMATIC FL STARTED <<<");
      delay(1000);
    }
    // 保持 BLE 轮询，虽然还没开始业务
    BLE.poll();
    return;
  }

  // 2. 自动循环逻辑
  if (current_round < TOTAL_ROUNDS) {
    // 执行一轮完整的任务 (训练 -> 等待连接 -> 接收更新)
    do_round_task();
    
    // 稍微延时，给 Central 断开连接的时间
    delay(500); 
  } else {
    Serial.println("All rounds finished. Stopping.");
    while(1);
  }
  
  // 保持 BLE 活跃 (关键)
  BLE.poll();
}