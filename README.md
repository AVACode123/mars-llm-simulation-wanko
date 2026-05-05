# mars-llm-simulation-wanko
LLM-based multi-agent Mars exploration simulation

# 🚀 Mars LLM Simulation

LLMエージェントによる火星探査シミュレーション。
チームに犬（ワンコ、Pochi）を入れた場合と化学者（Joe）を入れた場合の
行動・ストレス・探査効率を比較。

環境設定、結果、解析、考察などはAnalysis内のPPTXファイルを参照ください。


## 実験概要
- 条件：Dog条件 vs Chemist条件
- 試行数：各10回（探索3回→検証10回）
- エージェント数：9名/チーム
- LLM：GPT-4o-mini

## 主な発見
- Dogチームはストレスが低い傾向（d=0.57）
- 小サンプル（n=3）では効果量を過大推定（d=1.41→0.57）
- 固着パターンはLLMエージェントの普遍的特性

## 実行方法
bash

pip install -r requirements.txt

python main.py

