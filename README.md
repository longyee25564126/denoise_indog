# BitPlaneFormer
### 基於位元平面的 Transformer 影像去噪系統

---

BitPlaneFormer 是一個結合 **位元平面拆解（Bit-plane Decomposition）** 與 **Transformer 架構** 的影像去噪研究專案。

本專案的核心思想在於：

> 將影像依位元重要性拆分為 **高位元（MSB）** 與 **低位元（LSB）**，  
> 分別對應影像的結構資訊與雜訊特性，  
> 並透過 **遮罩引導的殘差學習（Mask-guided Residual Learning）**  
> 來提升整體去噪效果。

---

## 專案核心概念

### MSB（Most Significant Bits）
- 保留影像的主要結構與語意資訊  
- 作為模型穩定且可靠的 **結構先驗（Structural Prior）**

---

### LSB（Least Significant Bits）
- 含有較多雜訊與高頻變化  
- 適合用來建模噪聲分佈與不確定性

---

### Transformer-based Feature Fusion
- 使用 Transformer 建立 MSB 與 LSB 特徵之間的全域關聯  
- 強化結構資訊對雜訊抑制的引導能力

---

### Residual Learning（殘差學習）
- 模型不直接預測乾淨影像  
- 而是學習輸入影像中的 **噪聲殘差（Noise Residual）**

---

## 系統流程概述

BitPlaneFormer 的整體流程可概括為以下階段：

1. **原始影像輸入**  
   將輸入影像轉換為位元表示形式

2. **位元平面拆解**  
   拆分為高位元（MSB）與低位元（LSB）平面

3. **特徵提取**  
   使用 Encoder 萃取 MSB 的結構特徵

4. **遮罩（Mask）預測**  
   針對 LSB 預測雜訊相關的遮罩資訊  
   引導模型聚焦於高噪聲區域

5. **特徵融合**  
   透過 Transformer 融合 MSB 與 LSB 特徵

6. **殘差預測（Patch Domain）**  
   在 patch / token 空間中預測噪聲殘差

7. **殘差影像重建**  
   將殘差 token 還原至影像空間

8. **最終輸出**  
   將預測殘差自輸入影像中扣除  
   得到最終去噪結果

---

## 專案結構說明

### models/
模型核心架構與主要網路模組

- **bitplane_former_v1.py**  
  模型主體與前向流程定義

- **msb_encoder.py**  
  高位元（MSB）結構特徵編碼器

- **lsb_mask_head.py**  
  低位元（LSB）雜訊遮罩預測模組

- **denoise_decoder.py**  
  特徵融合與殘差影像還原模組

---

### datasets/
資料前處理與位元操作相關模組

- **bitplane_utils.py**  
  位元平面提取與資料型別轉換工具

- **external_adapter.py**  
  外部資料集格式轉接模組

---

### tools/
輔助分析與視覺化工具

- **run_infer_visualize.py**  
  推論流程與去噪結果視覺化

- **compute_bitflip_stats.py**  
  各位元平面雜訊翻轉率分析

---

## 快速上手

### 1️. 模型訓練
透過設定檔啟動訓練流程，  
進行 BitPlaneFormer 的端到端學習。

---

### 2️. 模型評估
在測試資料集上進行定量評估，常用指標包含：

- **PSNR / SSIM**  
  衡量像素層級與結構重建品質

- **LPIPS**  
  衡量人眼感知的視覺相似度

評估結果將輸出為統計報表與視覺化對比影像。

---

## 核心技術：殘差學習（Residual Learning）

本專案採用殘差學習策略：

- 模型專注於學習輸入影像中的 **噪聲成分**
- 而非直接重建乾淨影像本身

實際推論時，去噪結果的產生方式如下：

denoised_image = input_image - predicted_residual


此設計有助於：
- 提升訓練穩定性
- 強化對雜訊分佈的建模能力

---


