import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="eval_results.csv")
    parser.add_argument("--output-dir", default="analysis_report")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    os.makedirs(args.output_dir, exist_ok=True)

    # Performance gains
    df['Delta_PSNR'] = df['PSNR_Denoised'] - df['PSNR_Noisy']
    df['Delta_LPIPS'] = df['LPIPS_Noisy'] - df['LPIPS_Denoised']

    # 1. Summary Statistics
    summary = df.mean(numeric_only=True)
    summary.to_csv(os.path.join(args.output_dir, "overall_summary.csv"))
    print("\n--- Average Metrics ---\n", summary)

    # 2. Group by Label
    if 'Label' in df.columns:
        cat_stats = df.groupby('Label').mean(numeric_only=True)
        cat_stats.to_csv(os.path.join(args.output_dir, "category_summary.csv"))
        print("\n--- Metrics by Label ---\n", cat_stats[['PSNR_Denoised', 'Delta_PSNR', 'LPIPS_Denoised']])

    # 3. Plots
    sns.set_theme(style="whitegrid")
    
    # Distribution Plot
    plt.figure(figsize=(8, 5))
    sns.histplot(df['Delta_PSNR'], kde=True)
    plt.title("PSNR Gain Distribution")
    plt.savefig(os.path.join(args.output_dir, "psnr_gain_dist.png"))

    # Correlation Plot
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x='PSNR_Denoised', y='LPIPS_Denoised', hue='Label' if 'Label' in df.columns else None)
    plt.title("PSNR vs LPIPS Correlation")
    plt.savefig(os.path.join(args.output_dir, "psnr_lpips_corr.png"))

    print(f"\nAnalysis complete. Results in {args.output_dir}")

if __name__ == "__main__": 
    main()