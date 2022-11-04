#pragma once
#define GPU_NAME "Tesla K40c"
#define GPU_VERSION_MAJOR 3
#define GPU_VERSION_MINOR 5
#define RT_VERSION 5050
#define DRV_VERSION 6050

#define DEF_THREADS_PER_BLOCK 960
#define UPDATE_THREADS_PER_BLOCK 1024
#define HCD_THREADS_PER_BLOCK 480
#define COPY_INV_THREADS_PER_BLOCK 512
#define STORE_INV_THREADS_PER_BLOCK 416
#define GEP_INV_THREADS_PER_BLOCK 480
static const char *TUNING_PARAMETERS = "DEF_THREADS_PER_BLOCK 960\nUPDATE_THREADS_PER_BLOCK 1024\nHCD_THREADS_PER_BLOCK 480\nCOPY_INV_THREADS_PER_BLOCK 512\nSTORE_INV_THREADS_PER_BLOCK 416\nGEP_INV_THREADS_PER_BLOCK 480\n";
