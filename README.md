# NASDAQ-100 CAGR 計算・予測ノートブック

このプロジェクトは、NASDAQ-100 の構成銘柄を取得し、年平均成長率（CAGR）を計算するとともに、機械学習モデルで将来の CAGR を予測する Jupyter Notebook を提供します。可視化や Excel 出力の機能もあります。

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

## 免責事項/Disclaimer
日本語

本プロジェクトは、教育・研究目的で公開されています。
本ツールおよびその出力結果は、金融商品取引法上の投資助言・勧誘・推奨を目的としたものではありません。
株価、経済指標、CAGRなどの分析結果は過去のデータに基づくものであり、
将来の投資成果やリターンを保証・示唆するものではありません。
利用者が本プログラムまたはその結果をもとに行ういかなる投資行為についても、
その判断および結果の責任はすべて利用者本人に帰属します。
本プロジェクトの作者は、本ソフトウェアの利用または結果に起因する
いかなる損害・損失についても一切の責任を負いません。
また、本ツールを第三者への投資勧誘・販売促進などの目的で利用することを禁止します。

English

This project is provided for educational and research purposes only.
It does not constitute investment advice, recommendation, or solicitation under any financial regulation (including the Financial Instruments and Exchange Act of Japan).
All analyses, predictions, or CAGR calculations are based on historical data, and
no guarantee or implication of future investment performance is made or intended.
Users are solely responsible for any financial decisions made based on the code or output of this project.
The author assumes no liability for any loss or damage resulting from its use.
This project must not be used for financial promotion, investment solicitation, or commercial advisory purposes.
