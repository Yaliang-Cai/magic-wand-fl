#ifndef FL_CONFIG_H
#define FL_CONFIG_H

// === BLE UUID 定义 ===
// 使用随机生成的 UUID，确保唯一性
const char* FL_SERVICE_UUID        = "19B10000-E8F2-537E-4F6C-D104768A1214";
const char* FL_WEIGHT_CHAR_UUID    = "19B10001-E8F2-537E-4F6C-D104768A1214"; // 传输权重
const char* FL_STATUS_CHAR_UUID    = "19B10002-E8F2-537E-4F6C-D104768A1214"; // 同步状态

// === 训练超参数 ===
const int EPOCHS_PER_ROUND = 2; // 每次聚合前，本地训练 10 轮
const int TOTAL_ROUNDS = 50;      // 总共进行 5 次联邦聚合

// === 传输协议配置 (Chunking) ===
// 单次传输 32 个 float (128 字节)，这在 BLE 限制内是安全的
const int CHUNK_SIZE_FLOATS = 32; 

// === 状态机定义 (握手信号) ===
// 基础状态
const int STATUS_TRAINING           = 0;
const int STATUS_READY_TO_AGGREGATE = 1;

// 第一阶段：Client (B) 发送数据给 Server (A)
const int STATUS_B_SENDING_CHUNK    = 10; // B: "我放好了一块数据"
const int STATUS_A_READ_ACK         = 11; // A: "我读完了，下一块"
const int STATUS_B_SENDING_COMPLETE = 12; // B: "我全部发完了"

// 第二阶段：Server (A) 发送数据回 Client (B)
const int STATUS_A_SENDING_CHUNK    = 20; // A: "我写回了一块数据"
const int STATUS_B_READ_ACK         = 21; // B: "我读完了，下一块"
const int STATUS_AGGREGATION_DONE   = 22; // A: "本轮结束，解散！"

#endif