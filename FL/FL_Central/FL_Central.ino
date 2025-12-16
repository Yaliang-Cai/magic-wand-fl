#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <TinyMLShield.h>
#include <ArduinoBLE.h>

// === 配置区域 ===
#define LEARNING_RATE 0.005
#define DATA_TYPE_FLOAT       
#define DEBUG 0

// 引入公共配置
#include "FL_Config.h"   

extern const int first_layer_input_cnt;
extern const int classes_cnt;

// 网络结构
static const unsigned int NN_def[] = {first_layer_input_cnt, 20, classes_cnt};

// ⚠️ 确保 data_A_new.h 是用脚本生成的：训练集缺2，但测试集包含全量数据
#include "data_A_new.h"      
#include "NN_functions.h"

// 全局变量
int iter_cnt = 0;        
int current_round = 0;   
int weights_count = 0;
// DATA_TYPE* WeightBiasPtr = NULL; // ⚠️ 已在 NN_functions.h 中定义，这里必须注释掉

// 🔥 新增：运行状态标志位
bool fl_running = false; 

// 1. 本地训练
void do_local_training() {
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
  Serial.print("[Local] Training Done. Total Epochs: "); Serial.println(iter_cnt);
}

// 2. 联邦聚合逻辑 (内存优化版)
bool do_federated_aggregation() {
  Serial.println("[BLE] Scanning for Peripheral (Node B)...");
  BLE.scanForUuid(FL_SERVICE_UUID);
  
  BLEDevice peripheral = BLE.available();
  unsigned long startScan = millis();
  
  while (!peripheral) {
    if (millis() - startScan > 10000) {
      BLE.stopScan();
      Serial.println("[BLE] Timeout. Retrying...");
      return false; 
    }
    peripheral = BLE.available();
  }
  BLE.stopScan();

  if (peripheral.connect()) {
    Serial.println("[BLE] Connected.");
    if (peripheral.discoverAttributes()) {
      BLECharacteristic wChar = peripheral.characteristic(FL_WEIGHT_CHAR_UUID);
      BLECharacteristic sChar = peripheral.characteristic(FL_STATUS_CHAR_UUID);
      
      // 等待 B 准备好
      byte status = 0;
      while (status != STATUS_READY_TO_AGGREGATE) {
        sChar.readValue(status);
        BLE.poll(); 
      }
      
      // 发送 ACK，通知 B 开始发送
      sChar.writeValue((byte)STATUS_A_READ_ACK);

      // --- A. 下载 B 的权重 ---
      Serial.println("[BLE] Downloading weights from B...");
      int recv_count = 0;
      bool receiving = true;
      while (receiving) {
        sChar.readValue(status);
        if (status == STATUS_B_SENDING_CHUNK) {
           int chunk_len = CHUNK_SIZE_FLOATS;
           if (recv_count + chunk_len > weights_count) chunk_len = weights_count - recv_count;
           
           wChar.readValue((uint8_t*)(WeightBiasPtr + recv_count), chunk_len * sizeof(float));
           recv_count += chunk_len;
           
           sChar.writeValue((byte)STATUS_A_READ_ACK);
        } else if (status == STATUS_B_SENDING_COMPLETE) {
           receiving = false;
        }
      }

      // --- B. 计算平均 (核心优化) ---
      Serial.println("[FedAvg] Aggregating using packUnpackVector(AVERAGE)...");
      packUnpackVector(2); // 2 = AVERAGE

      // --- C. 上传 Global Model ---
      Serial.println("[BLE] Uploading Global Model to B...");
      int sent_count = 0;
      while (sent_count < weights_count) {
         int chunk_len = CHUNK_SIZE_FLOATS;
         if (sent_count + chunk_len > weights_count) chunk_len = weights_count - sent_count;

         wChar.writeValue((uint8_t*)(WeightBiasPtr + sent_count), chunk_len * sizeof(float));
         sChar.writeValue((byte)STATUS_A_SENDING_CHUNK);

         while (1) {
            byte s = 0; sChar.readValue(s);
            if (s == STATUS_B_READ_ACK) break;
         }
         sent_count += chunk_len;
      }

      sChar.writeValue((byte)STATUS_AGGREGATION_DONE);
      
      // 给一点时间断开
      delay(500); 
      peripheral.disconnect();
      return true;
    }
    peripheral.disconnect();
  }
  return false;
}

void setup() {
  Serial.begin(9600);
  delay(5000); 

  initializeShield();
  srand(0);

  if (!BLE.begin()) { Serial.println("BLE failed!"); while (1); }

  weights_count = calcTotalWeightsBias();
  // 只分配一个缓冲区，省内存！
  WeightBiasPtr = (DATA_TYPE*) calloc(weights_count, sizeof(DATA_TYPE));
  setupNN(WeightBiasPtr);

  Serial.println("Arduino A (Central) Ready.");
  Serial.println("Press button ONCE to start FULL AUTOMATED FL."); // 提示按一次全自动
  printAccuracy();
}

void loop() {
  // 1. 等待启动信号 (只按一次)
  if (!fl_running) {
    if (readShieldButton()) {
      fl_running = true;
      Serial.println("\n>>> AUTOMATIC FL STARTED <<<");
      delay(1000); // 防抖
    }
    return; // 没按就一直在这等
  }

  // 2. 自动循环逻辑
  if (current_round < TOTAL_ROUNDS) {
    Serial.print("\n=== STARTING ROUND "); Serial.print(current_round + 1); 
    Serial.print(" / "); Serial.print(TOTAL_ROUNDS); Serial.println(" ===");
    
    // A. 本地训练
    do_local_training();
    
    // B. 联邦聚合
    if (do_federated_aggregation()) {
      Serial.println(">>> ROUND COMPLETE: Model Updated <<<");
      current_round++;
      
      // 打印详细测试结果
      Serial.print("[Metric] Detailed Accuracy after Aggregation: ");
      printDetailedTest(); // 确保这里用的是 printDetailedTest
      
    } else {
      Serial.println(">>> ROUND FAILED: Connection Error. Retrying in 2s... <<<");
      delay(2000); // 失败稍微等一下再试，或者直接重试
    }
    
    // 每一轮结束后休息一下，让串口数据吐完，也给 B 一点时间准备
    delay(1000); 

  } else {
    // 3. 全部跑完
    Serial.println("\n=================================");
    Serial.println("       ALL ROUNDS FINISHED       ");
    Serial.println("=================================");
    printDetailedTest(); // 最后再测一次
    while(1); // 死循环结束
  }
}