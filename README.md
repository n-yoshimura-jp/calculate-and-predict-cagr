# NASDAQ-100 CAGR 計算・予測ノートブック

このプロジェクトは、NASDAQ-100 の構成銘柄を取得し、年平均成長率（CAGR）を計算するとともに、機械学習モデルで将来の CAGR を予測する Jupyter Notebook を提供します。可視化や Excel 出力を通じて、銘柄の成長性を直感的に把握できるようにしています。

## 主な機能

- Wikipedia から最新の NASDAQ-100 銘柄リストを取得
- `yfinance` を用いて各銘柄の株価履歴をダウンロードし、CAGR・上場年数・出来高などを算出
- 高成長かつ長期上場の銘柄を抽出し、Plotly でインタラクティブに可視化
- Random Forest、Gradient Boosting、Ridge、Linear Regression を組み合わせた回帰モデルで将来の CAGR を推定
- 中間結果と予測結果を Excel (`cagr_data.xlsx`, `cagr_prediction_data.xlsx`) に出力

## 必要条件

- Python 3.10 以上（仮想環境の利用を推奨）
- Jupyter Notebook または JupyterLab
- 詳細なパッケージ一覧は `requirements.txt` を参照してください（主な依存は `yfinance`、`pandas`、`plotly`、`scikit-learn`、`openpyxl`、`lxml`、`nbformat` など）。

## セットアップ

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

## 使い方

1. Jupyter を起動します。
   ```bash
   jupyter notebook
   ```
2. `src/calculate_and_predict_cagr_of_nasdaq100.ipynb` を開きます。
3. 上から順にセルを実行します。
   - 銘柄リストと株価履歴の取得にはインターネット接続が必要です。
   - 中間結果は `cagr_data.xlsx`、予測結果は `cagr_prediction_data.xlsx` に保存されます。
   - Plotly のグラフはブラウザ表示に対応した Jupyter フロントエンドでの実行を想定しています。

## 備考

- NASDAQ-100 全銘柄の全期間データを取得するため、実行には一定の時間がかかります。
- 機械学習モデルの特徴量やハイパーパラメータは Notebook 内で調整可能です。カスタマイズしてさまざまな分析を試してみてください。
