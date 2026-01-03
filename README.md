BitPlaneFormer: 基於位元平面的 Transformer 影像去噪系統
這是一個基於 位元平面（Bit-plane） 拆解與 Transformer 架構的影像去噪專案。本專案的核心概念是將影像拆解為高位元（MSB，保留結構）與低位元（LSB，捕捉雜訊），並透過 Transformer 進行特徵融合與遮罩（Mask）引導的殘差學習。

核心流程圖
程式碼片段

graph LR
    subgraph Main_Flow [主流程邏輯]
        direction TB
        A[1. 原始影像對] --> B[2. 提取位元平面]
        B --> C[3. 特徵提取]
        C --> D[4. 遮罩預測]
        D --> E[5. 特徵融合]
        E --> F[6. 預測殘差]
        F --> G[7. 影像重建]
        G --> H[8. 最終輸出]
    end

    subgraph Code_Reference [程式碼對應]
        direction TB
        CA(bitplane_dataset.py)
        CB(bitplane_utils.py)
        CC(msb_encoder.py)
        CD(lsb_mask_head.py)
        CE(denoise_decoder.py)
        CF(unpatchify 函式)
        CG(bitplane_former_v1.py)
        CH(eval.py)
    end

    A -.-> CA
    B -.-> CB
    C -.-> CC
    D -.-> CD
    E -.-> CE
    F -.-> CF
    G -.-> CG
    H -.->|效能與指標總評估| CH
專案結構說明
models/: 包含模型核心架構。

bitplane_former_v1.py: 模型主進入點。

msb_encoder.py: 處理高位元結構資訊。

lsb_mask_head.py: 預測雜訊遮罩 (Mask Prediction)。

denoise_decoder.py: 融合特徵並執行 unpatchify 還原殘差影像。

datasets/: 資料處理與位元操作。

bitplane_utils.py: 包含 to_uint8 與位元平面提取工具。

external_adapter.py: 用於接軌外部資料集格式。

tools/: 實用的視覺化與分析工具。

run_infer_visualize.py: 執行推論並產生對比圖。

compute_bitflip_stats.py: 分析各個位元的雜訊翻轉率。

快速上手
1. 訓練模型
使用 YAML 設定檔啟動訓練：

Bash

python train_bitplane_former_v1.py --config configs/train_bitplane_v1.yaml
2. 定量指標評估 (Evaluation)
訓練完成後，使用 eval.py 針對測試集進行全面評估。此步驟會計算 PSNR、SSIM 與 LPIPS 指標，並產出統計報表。

Bash

python eval.py --root "/path/to/dataset" --checkpoint "/path/to/best.pth" --split "test" --save-images
PSNR/SSIM: 衡量像素與結構的重建精確度。

LPIPS: 衡量人眼感官的視覺相似度。

輸出: 產生 eval_results.csv 與視覺化對比圖。

核心技術：殘差學習
本專案採用 殘差學習 (Residual Learning)。模型不是直接預測乾淨影像，而是透過 LSB 資訊預測「雜訊殘差」，最後執行 y_hat = x - residual 來獲得去噪結果。