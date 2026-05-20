# ======================================================================================================
# Markov Analysis Suite Pro
# 功能：馬可夫鏈分析工具，支援警察巡邏
# 主要模組：轉移矩陣建構、穩態計算、回傳時間、互動模擬、視覺化
# ======================================================================================================

import streamlit as st
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
import streamlit.components.v1 as components
import time
import json

# ======================================================================================================
# 1. 視覺風格與 CSS 定義
#    使用 glassmorphism 設計語言，搭配 iOS 風格配色
# ======================================================================================================
def apply_custom_style() -> None:
    """注入全域 CSS，定義卡片、按鈕、分頁等元件的外觀。"""
    st.markdown("""
        <style>
        .stApp { background: linear-gradient(135deg, #f5f7fa 0%, #e4e7eb 100%); }
        .glass-card {
            background: rgba(255, 255, 255, 0.7) !important;
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border-radius: 24px !important;
            border: 1px solid rgba(255, 255, 255, 0.3) !important;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.07) !important;
            padding: 25px !important;
            margin-bottom: 20px !important;
            color: #1d1d1f;
        }
        .stMetric {
            background: rgba(255, 255, 255, 0.8) !important;
            padding: 20px !important;
            border-radius: 20px !important;
            box-shadow: 0 4px 12px rgba(0,0,0,0.03) !important;
            border: none !important;
        }
        .stButton>button {
            width: 100%; border-radius: 16px !important;
            background: linear-gradient(180deg, #007AFF, #005BBF) !important;
            color: white !important; font-weight: 600 !important;
            border: none !important; padding: 14px 20px !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 15px rgba(0, 122, 255, 0.3) !important;
        }
        .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(0, 122, 255, 0.4) !important; }
        div.stButton > div.st-emotion-cache-micr9v > button {
            background: linear-gradient(180deg, #FF3B30, #D70015) !important;
            box-shadow: 0 4px 15px rgba(255, 59, 48, 0.3) !important;
        }
        .mode-selector {
            background: rgba(255, 255, 255, 0.5); backdrop-filter: blur(15px);
            padding: 30px; border-radius: 30px; border: 1px solid rgba(255, 255, 255, 0.4);
            margin-bottom: 30px; text-align: center; box-shadow: 0 10px 30px rgba(0,0,0,0.05);
        }
        .calc-box {
            background: rgba(255, 255, 255, 0.9); padding: 20px; border-radius: 20px;
            border-left: 8px solid #007AFF; box-shadow: 0 4px 12px rgba(0,0,0,0.05); margin: 15px 0;
        }
        .explain-box {
            background-color: #fffbe6; padding: 15px; border-radius: 16px;
            border: 1px solid #ffe58f; margin-top: 10px; color: #856404; font-size: 0.9rem;
        }
        .path-box {
            background: rgba(242, 242, 247, 0.8); padding: 15px; border-radius: 16px;
            border: 1px solid #d1d1d6; font-family: 'SF Mono', monospace; color: #3a3a3c;
        }
        .stTabs [data-baseweb="tab"] {
            background-color: transparent !important; border: none !important;
            border-radius: 12px 12px 0 0 !important; padding: 10px 20px !important;
        }
        .stTabs [aria-selected="true"] {
            background-color: rgba(255, 255, 255, 0.8) !important;
            border-bottom: 3px solid #007AFF !important; font-weight: bold !important;
        }
        </style>
    """, unsafe_allow_html=True)

# ======================================================================================================
# 2. 數學核心邏輯
# ======================================================================================================

def build_transition_matrix(n: int, edges_with_weights: list, allow_self_loop: bool = True):
    """
    建構馬可夫鏈轉移矩陣 P（行隨機矩陣：每列加總 = 1）。

    參數：
        n               : 節點總數
        edges_with_weights : [(u, v, weight), ...] 無向邊清單
        allow_self_loop : 若為 True，每個節點附加自環權重 1.0（警察巡邏模式）

    回傳：
        P   : shape (n, n) 的轉移矩陣，P[i][j] = 從節點 i+1 移動到 j+1 的機率
        adj : 鄰接字典 {node: [(neighbor, weight), ...]}
    """
    P = np.zeros((n, n))
    adj = {i: [] for i in range(1, n + 1)}

    # ── 建構無向鄰接表
    for u, v, w in edges_with_weights:
        if 1 <= u <= n and 1 <= v <= n:
            adj[u].append((v, w))
            adj[v].append((u, w))

    # ── 填入轉移機率：各邊權重 / 總權重（含自環）
    self_weight = 1.0 if allow_self_loop else 0.0
    for i in range(1, n + 1):
        neighbors = adj[i]
        total_weight = sum(w for _, w in neighbors) + self_weight
        if total_weight == 0:
            continue  # 孤立節點，跳過
        P[i-1, i-1] = self_weight / total_weight          # 自環機率
        for v, w in neighbors:
            P[i-1, v-1] = w / total_weight                # 移動到鄰居的機率

    return P, adj


def find_steady_state(P: np.ndarray, threshold: float):
    """
    以冪迭代法 (power iteration) 求穩態分佈 π，
    滿足 π = π P（行向量左乘）。

    回傳：
        v             : 穩態機率向量 π，長度 n
        iteration     : 收斂所需迭代次數
        error_history : 每步最大誤差紀錄（用於收斂曲線）
    """
    n = P.shape[0]
    if n == 0:
        return np.array([]), 0, []

    v = np.zeros(n)
    v[0] = 1.0          # 初始分佈：全部機率集中在節點 1
    error_history = []
    iteration = 0

    while True:
        v_next = np.dot(v, P)                          # 一步更新：v^(t+1) = v^(t) · P
        error = np.max(np.abs(v_next - v))             # 最大分量誤差
        error_history.append(error)
        if error < threshold or iteration > 10000:
            break
        v = v_next
        iteration += 1

    return v, iteration, error_history


def compute_return_times(steady_v: np.ndarray) -> np.ndarray:
    """
    計算各節點的理論平均回傳時間 (Mean First Return Time)。

    根據馬可夫鏈定理：
        μ_i = 1 / π_i
    即穩態機率的倒數。π_i 越大，回傳越快。

    回傳：
        return_times : 長度 n 的陣列，第 i 個元素為節點 i+1 的平均回傳步數
                       若 π_i = 0（不可達狀態）則設為 ∞
    """
    with np.errstate(divide='ignore'):
        rt = np.where(steady_v > 1e-12, 1.0 / steady_v, np.inf)
    return rt


def get_convergence_history_fixed(P: np.ndarray, max_iters: int) -> list:
    """
    固定迭代次數，記錄每步的最大誤差，供收斂曲線繪製使用。
    """
    n = P.shape[0]
    if n == 0:
        return []
    v = np.zeros(n)
    v[0] = 1.0
    error_history = []
    for _ in range(max_iters):
        v_next = np.dot(v, P)
        error = np.max(np.abs(v_next - v))
        error_history.append(error)
        v = v_next
    return error_history


# ======================================================================================================
# 3. 視覺化組件
# ======================================================================================================

def create_interactive_graph(n: int, edges_with_weights: list,
                              steady_v=None, return_times=None,
                              fixed_pos=None, label_prefix: str = "位置"):
    """
    使用 PyVis 產生互動式網路圖。
    節點顏色依穩態機率深淺染色（越紅 = 機率越高）。
    Hover tooltip 顯示：穩態機率、平均回傳時間。
    """
    net = Network(height="500px", width="100%", bgcolor="#ffffff", font_color="black")
    if fixed_pos:
        net.set_options('{"physics":{"enabled":false},"nodes":{"font":{"size":16}}}')
    else:
        net.barnes_hut()

    for i in range(1, n + 1):
        color = "#ADD8E6"
        tooltip = f"{label_prefix} {i}"

        if steady_v is not None and len(steady_v) >= i:
            pi = steady_v[i-1]
            intensity = int(pi * 255)
            color = f"rgb(255, {255 - min(intensity, 255)}, {255 - min(intensity, 255)})"
            tooltip += f"\n穩態機率 π: {pi:.4f} ({pi*100:.2f}%)"

        if return_times is not None and len(return_times) >= i:
            rt = return_times[i-1]
            tooltip += f"\n理論回傳步數 μ: {rt:.2f} 步" if not np.isinf(rt) else "\n理論回傳步數: ∞"

        if fixed_pos:
            pos = fixed_pos.get(i, (0, 0))
            net.add_node(i, label=f"{label_prefix} {i}", color=color,
                         x=pos[0], y=pos[1], title=tooltip)
        else:
            net.add_node(i, label=f"{label_prefix} {i}", color=color, title=tooltip)

    for u, v, w in edges_with_weights:
        net.add_edge(u, v, value=w)

    net.save_graph("graph.html")
    return "graph.html"


def render_smooth_simulation(n_nodes: int, edges: list, P_matrix: np.ndarray,
                              start_node: int, speed: float,
                              label_prefix: str = "位置",
                              max_steps: int = 500,
                              steady_v: np.ndarray = None,
                              return_times: np.ndarray = None):
    """
    渲染前端互動式隨機行走模擬器（純 HTML/JS，嵌入 vis.js）。

    新增功能：
      - ▶ / ⏸ / ↺ 即時控制
      - 節點熱度染色（依拜訪頻率）
      - 拜訪頻率即時排行
      - 實際拜訪比例 vs 理論穩態對比圖
      - 各節點實際平均回傳步數 vs 理論值（1/π_i）對比表
      - 路徑歷史紀錄（最近 40 步）
      - 節點 tooltip 顯示回傳時間資訊
    """
    P_json = P_matrix.tolist()

    # ── 準備理論穩態與回傳時間（傳遞給 JS）
    if steady_v is not None and len(steady_v) == n_nodes:
        steady_list = [float(x) for x in steady_v]
    else:
        steady_list = [1.0 / n_nodes] * n_nodes

    if return_times is not None and len(return_times) == n_nodes:
        rt_list = [float(x) if not np.isinf(x) else -1 for x in return_times]
    else:
        rt_list = [-1] * n_nodes

    # ── 節點初始樣式
    nodes_js = [
        {"id": i, "label": f"{label_prefix}{i}",
         "color": {"background": "#ADD8E6", "border": "#7EB8D4"},
         "title": f"理論穩態: {steady_list[i-1]*100:.2f}%  |  理論回傳: {rt_list[i-1]:.2f} 步" if rt_list[i-1] > 0 else f"理論穩態: {steady_list[i-1]*100:.2f}%"}
        for i in range(1, n_nodes + 1)
    ]

    # ── 邊樣式
    edges_js = [
        {"from": u, "to": v, "width": max(1, w * 2), "color": {"color": "#C8C8D0"}}
        for u, v, w in edges
    ]

    html_content = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  /* ── 全域重設 ── */
  * {{ box-sizing: border-box; margin: 0; padding: 0;
       font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }}
  body {{ background: #f0f2f5; }}
  #wrap {{ display: flex; flex-direction: column; height: 900px; gap: 8px; padding: 10px; }}

  /* ── 頂部統計列 ── */
  #statsbar {{ display: flex; gap: 8px; flex-shrink: 0; }}
  .stat-pill {{
    flex: 1; background: white; border-radius: 14px;
    padding: 10px 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    display: flex; flex-direction: column; align-items: center;
  }}
  .stat-label {{ font-size: 9px; color: #8e8e93; font-weight: 700;
                 letter-spacing: 0.6px; text-transform: uppercase; }}
  .stat-value {{ font-size: 20px; font-weight: 700; color: #1d1d1f; margin-top: 2px; }}
  .stat-sub   {{ font-size: 9px; color: #8e8e93; margin-top: 1px; }}

  /* ── 主區：圖 + 側面板 ── */
  #main {{ display: flex; gap: 8px; flex: 1; min-height: 0; }}
  #graph-wrap {{ flex: 1; background: white; border-radius: 20px;
                 box-shadow: 0 4px 16px rgba(0,0,0,0.06); overflow: hidden; }}
  #mynetwork {{ width: 100%; height: 100%; }}

  /* ── 側面板 ── */
  #sidepanel {{ width: 190px; display: flex; flex-direction: column; gap: 8px; flex-shrink: 0; }}
  .panel-box {{ background: white; border-radius: 16px; padding: 12px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.06); }}
  .panel-title {{ font-size: 9px; font-weight: 700; color: #8e8e93;
                  letter-spacing: 0.6px; text-transform: uppercase; margin-bottom: 8px; }}

  /* ── 控制按鈕 ── */
  #controls {{ display: flex; gap: 5px; }}
  .ctrl-btn {{
    flex: 1; border: none; border-radius: 11px; padding: 9px 4px;
    font-size: 12px; font-weight: 700; cursor: pointer; transition: all 0.2s;
  }}
  #btn-play  {{ background: linear-gradient(180deg,#34C759,#248A3D); color:white; }}
  #btn-pause {{ background: linear-gradient(180deg,#FF9500,#CC7700); color:white; }}
  #btn-reset {{ background: linear-gradient(180deg,#FF3B30,#CC2218); color:white; }}
  .ctrl-btn:hover {{ transform: translateY(-1px); opacity: 0.9; }}
  .ctrl-btn:disabled {{ opacity: 0.35; cursor: default; transform: none; }}

  /* ── 速度滑桿 ── */
  #speed-slider {{ width: 100%; accent-color: #007AFF; margin-top: 4px; }}
  .speed-label {{ font-size: 10px; color: #8e8e93; text-align: center; }}

  /* ── 拜訪頻率列 ── */
  .freq-row {{ display: flex; align-items: center; gap: 5px; margin-bottom: 5px; }}
  .freq-label {{ font-size: 10px; color: #3a3a3c; width: 36px; flex-shrink: 0; font-weight: 600; }}
  .freq-bar-wrap {{ flex: 1; background: #f2f2f7; border-radius: 4px; height: 7px; overflow: hidden; }}
  .freq-bar {{ height: 100%; border-radius: 4px;
               background: linear-gradient(90deg,#007AFF,#5AC8FA); transition: width 0.4s ease; }}
  .freq-pct {{ font-size: 9px; color: #8e8e93; width: 30px; text-align: right; flex-shrink: 0; }}

  /* ── 路徑歷史 ── */
  #path-scroll {{ max-height: 36px; overflow-x: auto; overflow-y: hidden;
                  white-space: nowrap; font-size: 11px; }}
  #path-scroll::-webkit-scrollbar {{ height: 3px; }}
  #path-scroll::-webkit-scrollbar-thumb {{ background: #c7c7cc; border-radius: 2px; }}
  .path-step {{ display: inline-block; background: #f2f2f7; border-radius: 6px;
                padding: 1px 6px; margin: 1px; font-weight: 600; font-size: 10px; color: #3a3a3c; }}
  .path-step.current {{ background: #FFD60A; color: #1d1d1f; }}

  /* ── 底部雙圖區 ── */
  #bottom-area {{ display: flex; gap: 8px; flex-shrink: 0; }}

  /* ── 機率比較圖 ── */
  #probchart {{ background: white; border-radius: 16px; padding: 12px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.06); flex: 1; }}
  #barchart {{ display: flex; align-items: flex-end; gap: 3px; height: 55px; margin-top: 6px; }}
  .bar-col {{ flex: 1; display: flex; flex-direction: column; align-items: center; gap: 1px; }}
  .bar-fill {{ width: 100%; border-radius: 3px 3px 0 0; transition: height 0.5s ease; min-height: 2px; }}
  .bar-lbl  {{ font-size: 8px; color: #8e8e93; }}

  /* ── 回傳時間對比表 ── */
  #return-table {{ background: white; border-radius: 16px; padding: 12px;
                   box-shadow: 0 2px 8px rgba(0,0,0,0.06); flex: 1; overflow: auto; }}
  #return-table table {{ width: 100%; border-collapse: collapse; font-size: 10px; }}
  #return-table th {{ color: #8e8e93; font-weight: 700; text-align: center;
                      padding: 3px 5px; border-bottom: 1px solid #f2f2f7;
                      letter-spacing: 0.3px; }}
  #return-table td {{ text-align: center; padding: 3px 5px;
                      border-bottom: 1px solid #f8f8fa; color: #1d1d1f; }}
  .rt-good {{ color: #34C759; font-weight: 700; }}  /* 實際接近理論 */
  .rt-warn {{ color: #FF9500; font-weight: 700; }}  /* 偏差較大 */

  /* ── 圖例 ── */
  .legend-row {{ display: flex; align-items: center; gap: 5px;
                 margin-bottom: 4px; font-size: 10px; color: #3a3a3c; }}
  .legend-dot {{ width: 11px; height: 11px; border-radius: 50%; flex-shrink: 0; }}
</style>
</head>
<body>
<div id="wrap">

  <!-- ① 頂部統計列 -->
  <div id="statsbar">
    <div class="stat-pill">
      <div class="stat-label">步數</div>
      <div class="stat-value" id="stat-steps">0</div>
    </div>
    <div class="stat-pill">
      <div class="stat-label">目前位置</div>
      <div class="stat-value" id="stat-pos">{start_node}</div>
    </div>
    <div class="stat-pill">
      <div class="stat-label">最常拜訪</div>
      <div class="stat-value" id="stat-top">—</div>
      <div class="stat-sub" id="stat-top-sub"></div>
    </div>
    <div class="stat-pill">
      <div class="stat-label">平均回傳（目前節點）</div>
      <div class="stat-value" id="stat-rt">—</div>
      <div class="stat-sub" id="stat-rt-theory"></div>
    </div>
    <div class="stat-pill">
      <div class="stat-label">狀態</div>
      <div class="stat-value" id="stat-status" style="font-size:13px;color:#34C759;">就緒</div>
    </div>
  </div>

  <!-- ② 主區：網路圖 + 側面板 -->
  <div id="main">
    <div id="graph-wrap"><div id="mynetwork"></div></div>

    <div id="sidepanel">
      <!-- 控制面板 -->
      <div class="panel-box">
        <div class="panel-title">控制</div>
        <div id="controls">
          <button class="ctrl-btn" id="btn-play"  onclick="startSim()">▶ 開始</button>
          <button class="ctrl-btn" id="btn-pause" onclick="pauseSim()" disabled>⏸</button>
          <button class="ctrl-btn" id="btn-reset" onclick="resetSim()">↺</button>
        </div>
        <div style="margin-top:10px;">
          <div class="panel-title">速度調整</div>
          <input type="range" id="speed-slider" min="100" max="2000"
                 value="{int(speed*1000)}" oninput="updateSpeed(this.value)">
          <div class="speed-label" id="speed-display">{speed:.1f} 秒/步</div>
        </div>
      </div>

      <!-- 拜訪頻率排行 -->
      <div class="panel-box" style="flex:1;overflow:hidden;display:flex;flex-direction:column;">
        <div class="panel-title">拜訪頻率</div>
        <div id="freq-bars" style="flex:1;overflow-y:auto;"></div>
      </div>

      <!-- 圖例 -->
      <div class="panel-box">
        <div class="panel-title">圖例</div>
        <div class="legend-row">
          <div class="legend-dot" style="background:#FFD60A;border:2px solid #FFB800;"></div>目前位置
        </div>
        <div class="legend-row">
          <div class="legend-dot" style="background:#007AFF;border:2px solid #005BBF;"></div>上一步
        </div>
        <div class="legend-row">
          <div class="legend-dot" style="background:#ADD8E6;border:2px solid #7EB8D4;"></div>一般節點
        </div>
        <div class="legend-row">
          <div class="legend-dot" style="background:linear-gradient(135deg,#FF6B6B,#FFD60A);"></div>高頻（熱度）
        </div>
      </div>
    </div>
  </div>

  <!-- ③ 路徑歷史 -->
  <div class="panel-box" style="flex-shrink:0;">
    <div class="panel-title">路徑紀錄（最近 40 步）</div>
    <div id="path-scroll"></div>
  </div>

  <!-- ④ 底部：機率比較圖 + 回傳時間對比表 -->
  <div id="bottom-area">

    <!-- 機率對比柱狀圖 -->
    <div id="probchart">
      <div class="panel-title">實際拜訪比例（藍）vs 理論穩態（灰）</div>
      <div id="barchart"></div>
    </div>

    <!-- 回傳時間對比表 -->
    <div id="return-table">
      <div class="panel-title">各格回傳時間：實際 vs 理論（μ = 1/π）</div>
      <table>
        <thead>
          <tr>
            <th>節點</th>
            <th>π（理論）</th>
            <th>μ = 1/π（理論）</th>
            <th>實際平均回傳</th>
            <th>差異</th>
            <th>樣本數</th>
          </tr>
        </thead>
        <tbody id="return-tbody"></tbody>
      </table>
    </div>

  </div>
</div>

<!-- vis.js CDN -->
<script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
<script>
// ══════════════════════════════════════════════════════
// 常數與初始資料（由 Python 注入）
// ══════════════════════════════════════════════════════
const N          = {n_nodes};           // 節點總數
const P          = {P_json};            // 轉移矩陣（行=起點，列=終點）
const startNode  = {start_node};        // 使用者設定的起始節點
const labelPrefix= "{label_prefix}";   // 節點標籤前綴
const MAX_STEPS  = {max_steps};         // 模擬步數上限
const STEADY     = {steady_list};       // 理論穩態機率 π_i
const THEORY_RT  = {rt_list};          // 理論回傳時間 1/π_i（-1 表示無窮大）

// ══════════════════════════════════════════════════════
// 建構 vis.js 網路圖
// ══════════════════════════════════════════════════════
const visNodes = new vis.DataSet({nodes_js});
const visEdges = new vis.DataSet({edges_js});
const network  = new vis.Network(
  document.getElementById('mynetwork'),
  {{ nodes: visNodes, edges: visEdges }},
  {{
    nodes: {{
      shape: 'circle', size: 28,
      font: {{ size: 14, face: '-apple-system', bold: {{ color: '#1d1d1f' }} }},
      borderWidth: 2,
      shadow: {{ enabled: true, size: 6, x: 0, y: 2 }}
    }},
    edges: {{ smooth: {{ type: 'continuous' }}, color: {{ inherit: false }}, selectionWidth: 0 }},
    physics: {{ enabled: true, barnesHut: {{ gravitationalConstant: -2500, centralGravity: 0.3, springLength: 160 }} }},
    interaction: {{ hover: true, tooltipDelay: 80 }}
  }}
);

// ══════════════════════════════════════════════════════
// 模擬狀態變數
// ══════════════════════════════════════════════════════
let currentNode  = startNode;   // 目前所在節點
let prevNode     = -1;          // 上一步節點（用於染色還原）
let stepCount    = 0;           // 總步數計數器
let running      = false;       // 模擬是否執行中
let intervalId   = null;        // setTimeout 的 ID
let simSpeed     = {int(speed*1000)}; // 毫秒/步

// ── 拜訪統計
let visitCount   = new Array(N + 1).fill(0);  // visitCount[i] = 節點 i 被拜訪的次數

// ── 回傳時間追蹤：lastLeftStep[i] = 最後一次離開節點 i 的步數
//    當再次抵達節點 i 時，計算回傳間隔並累積
let lastLeftStep   = new Array(N + 1).fill(-1);
let returnSamples  = Array.from({{length: N + 1}}, () => []);  // 各節點的回傳時間樣本列表

// ── 路徑歷史（最近 40 步）
let pathHistory  = [];

// ── 邊快速查詢表：edgeMap["u-v"] = edgeId
let edgeMap = {{}};
visEdges.getIds().forEach(id => {{
  const e = visEdges.get(id);
  edgeMap[`${{e.from}}-${{e.to}}`] = id;
  edgeMap[`${{e.to}}-${{e.from}}`] = id;
}});

// ── 初始化：標記起點
visitCount[startNode]++;
pathHistory.push(startNode);
highlightNode(startNode, 'current');
renderAll();

// ══════════════════════════════════════════════════════
// 節點顏色函式
// ══════════════════════════════════════════════════════

/** 依拜訪頻率比例計算熱度顏色（低→藍，高→橘紅） */
function heatColor(ratio) {{
  const r = Math.round(173 + (255 - 173) * ratio);
  const g = Math.round(216 - 216 * ratio * 0.75);
  const b = Math.round(230 - 230 * ratio * 0.85);
  return `rgb(${{r}},${{g}},${{b}})`;
}}

/** 更新節點的顏色與大小 */
function highlightNode(nodeId, type) {{
  if (type === 'current') {{
    // 目前位置：金黃色 + 放大
    visNodes.update({{id: nodeId, color: {{background:'#FFD60A', border:'#FFB800'}},
                       shadow: {{enabled:true, size:12}}, size: 34}});
  }} else if (type === 'prev') {{
    // 上一步：藍色（短暫停留後切換為熱度色）
    visNodes.update({{id: nodeId, color: {{background:'#007AFF', border:'#005BBF'}},
                       shadow: {{enabled:true, size:6}}, size: 28}});
  }} else if (type === 'heat') {{
    // 熱度染色：依累積拜訪比例
    const maxV = Math.max(...visitCount.slice(1)) || 1;
    const ratio = visitCount[nodeId] / maxV;
    const bg = heatColor(ratio);
    visNodes.update({{id: nodeId, color: {{background: bg, border:'#7EB8D4'}},
                       shadow: {{enabled:true, size:4}}, size: 28}});
  }} else {{
    // 恢復預設
    visNodes.update({{id: nodeId, color: {{background:'#ADD8E6', border:'#7EB8D4'}},
                       shadow: {{enabled:false}}, size: 28}});
  }}
}}

// ══════════════════════════════════════════════════════
// 核心：單步移動邏輯
// ══════════════════════════════════════════════════════

async function step() {{
  if (!running) return;

  // ── 依轉移矩陣抽樣下一個節點
  const probs = P[currentNode - 1];
  let rand = Math.random(), nextNode = 1, cum = 0;
  for (let i = 0; i < probs.length; i++) {{
    cum += probs[i];
    if (rand <= cum) {{ nextNode = i + 1; break; }}
  }}

  // ── 邊動畫：閃藍 → 還原
  const edgeId = edgeMap[`${{currentNode}}-${{nextNode}}`];
  if (edgeId !== undefined) {{
    visEdges.update({{id: edgeId, color: {{color:'#007AFF'}}, width: 5}});
    await delay(simSpeed * 0.35);
    if (!running) {{ visEdges.update({{id: edgeId, color: {{color:'#C8C8D0'}}, width: 2}}); return; }}
    visEdges.update({{id: edgeId, color: {{color:'#C8C8D0'}}, width: 2}});
  }} else {{
    await delay(simSpeed * 0.35);
  }}

  // ── 節點顏色更新
  if (prevNode > 0) highlightNode(prevNode, 'heat');  // 前前步 → 熱度色
  highlightNode(currentNode, 'prev');                 // 前一步 → 藍色
  highlightNode(nextNode, 'current');                 // 新位置 → 金黃

  // ── 回傳時間追蹤
  //    離開 currentNode：記錄離開步數
  if (lastLeftStep[currentNode] < 0) {{
    lastLeftStep[currentNode] = stepCount;  // 第一次離開
  }}
  //    抵達 nextNode：若之前曾離開過，計算回傳間隔
  if (lastLeftStep[nextNode] >= 0) {{
    const interval = stepCount - lastLeftStep[nextNode] + 1;  // +1 因為這一步才到達
    returnSamples[nextNode].push(interval);
  }}
  //    抵達後立刻標記「已在此節點，尚未離開」
  lastLeftStep[nextNode] = -1;   // 重置：等下一次離開再記
  // 若下一步不在同節點才更新 lastLeftStep（下一步再處理）

  // ── 更新狀態
  prevNode = currentNode;
  currentNode = nextNode;
  stepCount++;
  visitCount[currentNode]++;
  pathHistory.push(currentNode);
  if (pathHistory.length > 40) pathHistory.shift();

  renderAll();

  // ── 步數上限檢查
  if (stepCount >= MAX_STEPS) {{
    pauseSim();
    document.getElementById('stat-status').textContent = '完成';
    document.getElementById('stat-status').style.color = '#FF9500';
  }}
}}

function delay(ms) {{ return new Promise(r => setTimeout(r, ms)); }}

// ══════════════════════════════════════════════════════
// 模擬控制函式
// ══════════════════════════════════════════════════════

function startSim() {{
  if (running) return;
  running = true;
  document.getElementById('btn-play').disabled  = true;
  document.getElementById('btn-pause').disabled = false;
  document.getElementById('stat-status').textContent = '執行中';
  document.getElementById('stat-status').style.color = '#34C759';
  scheduleNext();
}}

/** 遞迴排程：上一步完成後，等待剩餘時間再執行下一步 */
function scheduleNext() {{
  if (!running) return;
  step().then(() => {{
    if (running) intervalId = setTimeout(scheduleNext, simSpeed * 0.65);
  }});
}}

function pauseSim() {{
  running = false;
  clearTimeout(intervalId);
  document.getElementById('btn-play').disabled  = false;
  document.getElementById('btn-pause').disabled = true;
  document.getElementById('stat-status').textContent = '暫停';
  document.getElementById('stat-status').style.color = '#FF9500';
}}

function resetSim() {{
  pauseSim();
  // 重置所有狀態
  currentNode   = startNode;
  prevNode      = -1;
  stepCount     = 0;
  visitCount    = new Array(N + 1).fill(0);
  lastLeftStep  = new Array(N + 1).fill(-1);
  returnSamples = Array.from({{length: N + 1}}, () => []);
  pathHistory   = [startNode];
  visitCount[startNode]++;
  // 恢復所有節點顏色
  for (let i = 1; i <= N; i++) highlightNode(i, 'none');
  highlightNode(startNode, 'current');
  renderAll();
  document.getElementById('stat-status').textContent = '就緒';
  document.getElementById('stat-status').style.color = '#34C759';
}}

/** 速度滑桿回呼 */
function updateSpeed(val) {{
  simSpeed = parseInt(val);
  document.getElementById('speed-display').textContent = (simSpeed / 1000).toFixed(1) + ' 秒/步';
}}

// ══════════════════════════════════════════════════════
// UI 渲染函式
// ══════════════════════════════════════════════════════

/** 一次性更新所有 UI 區塊 */
function renderAll() {{
  updateTopStats();
  renderFreqBars();
  renderProbChart();
  renderReturnTable();
  renderPathHistory();
}}

/** 更新頂部統計數字 */
function updateTopStats() {{
  document.getElementById('stat-steps').textContent = stepCount;
  document.getElementById('stat-pos').textContent   = labelPrefix + currentNode;

  // 最常拜訪節點
  const maxV   = Math.max(...visitCount.slice(1));
  const topIdx = visitCount.indexOf(maxV);
  const total  = visitCount.slice(1).reduce((a,b)=>a+b,0) || 1;
  document.getElementById('stat-top').textContent     = labelPrefix + topIdx;
  document.getElementById('stat-top-sub').textContent = (maxV/total*100).toFixed(1) + '%';

  // 目前節點的回傳時間資訊
  const rtSamples = returnSamples[currentNode];
  if (rtSamples.length > 0) {{
    const actualRt = (rtSamples.reduce((a,b)=>a+b,0) / rtSamples.length).toFixed(1);
    document.getElementById('stat-rt').textContent = actualRt + ' 步';
  }} else {{
    document.getElementById('stat-rt').textContent = '—';
  }}
  const theory = THEORY_RT[currentNode - 1];
  document.getElementById('stat-rt-theory').textContent =
    theory > 0 ? `理論: ${{theory.toFixed(2)}} 步` : '理論: ∞';
}}

/** 渲染拜訪頻率橫條圖（右側面板） */
function renderFreqBars() {{
  const total = visitCount.slice(1).reduce((a,b)=>a+b,0) || 1;
  const maxV  = Math.max(...visitCount.slice(1)) || 1;
  let html = '';
  for (let i = 1; i <= N; i++) {{
    const pct = visitCount[i] / total;
    html += `<div class="freq-row">
      <div class="freq-label">${{labelPrefix}}${{i}}</div>
      <div class="freq-bar-wrap">
        <div class="freq-bar" style="width:${{(visitCount[i]/maxV*100).toFixed(1)}}%"></div>
      </div>
      <div class="freq-pct">${{(pct*100).toFixed(1)}}%</div>
    </div>`;
  }}
  document.getElementById('freq-bars').innerHTML = html;
}}

/** 渲染底部機率對比柱狀圖（藍=實際，灰=理論） */
function renderProbChart() {{
  const total     = visitCount.slice(1).reduce((a,b)=>a+b,0) || 1;
  const maxSteady = Math.max(...STEADY);
  let html = '';
  for (let i = 1; i <= N; i++) {{
    const actual = visitCount[i] / total;
    const theory = STEADY[i - 1];
    const aH = Math.max(2, Math.round(actual / maxSteady * 52));
    const tH = Math.max(2, Math.round(theory / maxSteady * 52));
    html += `<div class="bar-col">
      <div style="display:flex;align-items:flex-end;gap:1px;height:52px;">
        <div class="bar-fill" style="height:${{aH}}px;background:linear-gradient(180deg,#007AFF,#5AC8FA);width:48%;"></div>
        <div class="bar-fill" style="height:${{tH}}px;background:#d1d1d6;width:48%;"></div>
      </div>
      <div class="bar-lbl">${{i}}</div>
    </div>`;
  }}
  document.getElementById('barchart').innerHTML = html;
}}

/**
 * 渲染各格回傳時間對比表
 * 欄位：節點 | π理論 | μ=1/π(理論) | 實際平均回傳 | 差異% | 樣本數
 */
function renderReturnTable() {{
  let html = '';
  for (let i = 1; i <= N; i++) {{
    const pi      = STEADY[i - 1];
    const theory  = THEORY_RT[i - 1];   // -1 = 無窮大
    const samples = returnSamples[i];
    const nSamples= samples.length;
    const actual  = nSamples > 0 ? samples.reduce((a,b)=>a+b,0) / nSamples : null;

    // 理論值顯示
    const theoryStr = theory > 0 ? theory.toFixed(2) : '∞';

    // 實際值與差異
    let actualStr = '—';
    let diffStr   = '—';
    let diffClass = '';
    if (actual !== null && theory > 0) {{
      actualStr  = actual.toFixed(2);
      const diff = Math.abs((actual - theory) / theory * 100);
      diffStr    = diff.toFixed(1) + '%';
      diffClass  = diff < 20 ? 'rt-good' : 'rt-warn';  // 20% 內視為良好
    }} else if (actual !== null) {{
      actualStr = actual.toFixed(2);
    }}

    // 目前節點高亮行
    const rowStyle = i === currentNode ? 'background:#FFF9E6;' : '';

    html += `<tr style="${{rowStyle}}">
      <td><strong>${{labelPrefix}}${{i}}</strong></td>
      <td>${{(pi * 100).toFixed(2)}}%</td>
      <td>${{theoryStr}}</td>
      <td>${{actualStr}}</td>
      <td class="${{diffClass}}">${{diffStr}}</td>
      <td style="color:#8e8e93;">${{nSamples}}</td>
    </tr>`;
  }}
  document.getElementById('return-tbody').innerHTML = html;
}}

/** 渲染路徑歷史（最近 40 步，水平捲動） */
function renderPathHistory() {{
  const el = document.getElementById('path-scroll');
  el.innerHTML = pathHistory.map((n, idx) =>
    `<span class="path-step${{idx === pathHistory.length - 1 ? ' current' : ''}}">${{labelPrefix}}${{n}}</span>`
  ).join(' → ');
  el.scrollLeft = el.scrollWidth;  // 自動捲到最右側（最新）
}}
</script>
</body>
</html>"""
    return components.html(html_content, height=940)


# ======================================================================================================
# 4. Streamlit 主界面
# ======================================================================================================
st.set_page_config(page_title="Markov Analysis Suite Pro", layout="wide")
apply_custom_style()

# ── 模式選擇標題列
st.markdown('<div class="mode-selector">', unsafe_allow_html=True)
st.markdown("<h2 style='text-align:center;color:#1d1d1f;font-family:sans-serif;'>🛠️ 系統分析模式選擇</h2>",
            unsafe_allow_html=True)
mode = st.radio("請選擇您要分析的對象：",
                ["👮 交通警察巡邏 (Police Patrol)", ""],
                horizontal=True, label_visibility="collapsed")
st.markdown('</div>', unsafe_allow_html=True)

# ── 預設拓撲（5節點警察巡邏圖）
INITIAL_TOPO = {
    'n_nodes': 5,
    'edges': [(1,2,1.0),(2,4,1.0),(4,3,1.0),(3,1,1.0),
              (1,5,1.0),(2,5,1.0),(3,5,1.0),(4,5,1.0)],
    'fixed_pos': {1:(0,100),2:(100,100),3:(0,0),4:(100,0),5:(50,50)},
    'allow_self_loop': True
}
if 'topo_data' not in st.session_state:
    st.session_state.topo_data = INITIAL_TOPO.copy()

# ======================================================================================================
# 側邊欄：佈局設定
# ======================================================================================================
st.sidebar.header("⚙️ 配置中心")

if mode == "👮 交通警察巡邏 (Police Patrol)":
    with st.sidebar.expander("📍 佈局設定", expanded=True):
        layout_type = st.selectbox("選擇佈局", ["(5節點)佈局","3x4 網格","自定義網格","手動輸入"])

        if layout_type == "(5節點)佈局":
            st.session_state.topo_data = INITIAL_TOPO.copy()

        elif layout_type == "3x4 網格":
            # 建構 3×4 網格圖（橫向與縱向相鄰邊各一）
            edges = []
            for r in range(3):
                for c in range(4):
                    u = r * 4 + c + 1
                    if c < 3: edges.append((u, u + 1, 1.0))
                    if r < 2: edges.append((u, u + 4, 1.0))
            st.session_state.topo_data = {
                'n_nodes': 12, 'edges': edges, 'fixed_pos': None, 'allow_self_loop': True
            }

        elif layout_type == "自定義網格":
            rows = st.number_input("行數", 1, 10, 3)
            cols = st.number_input("列數", 1, 10, 4)
            edges = []
            for r in range(rows):
                for c in range(cols):
                    u = r * cols + c + 1
                    if c < cols - 1: edges.append((u, u + 1, 1.0))
                    if r < rows - 1: edges.append((u, u + cols, 1.0))
            st.session_state.topo_data = {
                'n_nodes': rows * cols, 'edges': edges, 'fixed_pos': None, 'allow_self_loop': True
            }

        elif layout_type == "手動輸入":
            raw_input = st.text_area("編輯關係清單 (u,v,w)", "1,2,1.0\n2,3,1.0\n3,1,1.0")
            temp_edges, curr_max = [], 0
            for line in raw_input.split('\n'):
                if line.strip():
                    try:
                        u, v, w = map(float, line.split(','))
                        temp_edges.append((int(u), int(v), w))
                        curr_max = max(curr_max, int(u), int(v))
                    except:
                        pass
            st.session_state.topo_data = {
                'n_nodes': max(curr_max, 2), 'edges': temp_edges,
                'fixed_pos': None, 'allow_self_loop': True
            }

elif mode == " (Mouse Maze)":
    with st.sidebar.expander("📍 迷宮設定", expanded=True):
        # 線性 8 格迷宮：節點 1-2-3-...-8，無自環
        edges    = [(i, i + 1, 1.0) for i in range(1, 8)]
        fixed_pos = {i: (i * 100, 0) for i in range(1, 9)}
        st.session_state.topo_data = {
            'n_nodes': 8, 'edges': edges, 'fixed_pos': fixed_pos, 'allow_self_loop': False
        }

with st.sidebar.expander("📈 數學精度設定", expanded=False):
    threshold = st.number_input("收斂閾值", value=0.000001, format="%.7f")

if st.sidebar.button("🔄 一鍵重置所有配置"):
    st.session_state.topo_data = INITIAL_TOPO.copy()
    st.rerun()

# ======================================================================================================
# 核心計算（每次重新渲染時執行）
# ======================================================================================================
n_nodes           = st.session_state.topo_data['n_nodes']
edges_with_weights= st.session_state.topo_data['edges']
fixed_pos         = st.session_state.topo_data['fixed_pos']
allow_self        = st.session_state.topo_data['allow_self_loop']
label_prefix      = "路口" if mode == "👮 交通警察巡邏 (Police Patrol)" else "位置"

# 建構轉移矩陣
P, adj = build_transition_matrix(n_nodes, edges_with_weights, allow_self_loop=allow_self)

# 求穩態分佈 π
steady_v, iters, _ = find_steady_state(P, threshold)

# 計算各節點理論平均回傳時間 μ_i = 1/π_i
return_times = compute_return_times(steady_v)

# ── 頂部三格指標
m_col1, m_col2, m_col3 = st.columns(3)
m_col1.metric("規模", f"{n_nodes} 處")
m_col2.metric("收斂次數", f"{iters} 次")
m_col3.metric("狀態", "穩定" if iters < 10000 else "未收斂")

# ======================================================================================================
# 分頁定義
# ======================================================================================================
tabs_list = [
    "🌐 互動拓撲圖", "⏱️ 隨機行走模擬", "📈 步數分佈演進",
    "📊 轉移矩陣", "📉 收斂趨勢", "🎯 穩定狀態與回傳時間",
    "📝 計算詳情", "📐 數學原理"
]
if mode == "🐁 8格迷宮老鼠 (Mouse Maze)":
    tabs_list.insert(4, "🧮 矩陣運算分析")

tab_objs = st.tabs(tabs_list)
tab_map  = {name: tab for name, tab in zip(tabs_list, tab_objs)}

# ======================================================================================================
# 分頁 1：互動拓撲圖
# ======================================================================================================
with tab_map["🌐 互動拓撲圖"]:
    st.subheader(f"{label_prefix}連接視覺化")
    st.caption("節點顏色由白→紅代表穩態機率由低到高；Hover 可查看機率與理論回傳時間。")
    # 傳入回傳時間，讓 tooltip 顯示 μ_i
    graph_html = create_interactive_graph(
        n_nodes, edges_with_weights, steady_v, return_times, fixed_pos, label_prefix
    )
    with open(graph_html, 'r', encoding='utf-8') as f:
        components.html(f.read(), height=550)

# ======================================================================================================
# 分頁 2：隨機行走模擬
# ======================================================================================================
with tab_map["⏱️ 隨機行走模擬"]:
    st.subheader("🚀 隨機行走模擬")
    col_ctrl, col_map_col = st.columns([1, 3])

    with col_ctrl:
        start_node_sim = st.number_input("設定起點", 1, n_nodes, 1)
        sim_speed      = st.slider("初始速度 (秒/步)", 0.1, 2.0, 0.8)
        max_steps_sim  = st.number_input("最大步數上限", 50, 5000, 500, step=50)
        st.markdown("""
        <div class="explain-box">
        <strong>✨ 功能說明：</strong><br>
        • ▶ / ⏸ / ↺ 即時控制，免重新載入<br>
        • 滑桿動態調速，執行中也可調整<br>
        • 節點熱度染色（拜訪越多越紅）<br>
        • 拜訪頻率即時排行（右側）<br>
        • 各格回傳時間：實際 vs 理論對比（底部表格）<br>
        • 實際拜訪比例 vs 理論穩態對比圖（底部）<br>
        • 路徑歷史紀錄（最近 40 步）
        </div>
        """, unsafe_allow_html=True)
        run_sim_btn = st.button("🎬 載入模擬")

    with col_map_col:
        if run_sim_btn:
            # 傳入穩態與回傳時間，讓模擬器顯示理論值對比
            render_smooth_simulation(
                n_nodes, edges_with_weights, P,
                start_node_sim, sim_speed, label_prefix,
                max_steps_sim, steady_v, return_times
            )
        else:
            st.info("設定好參數後，點擊左側「載入模擬」即可啟動。\n"
                    "模擬載入後可用畫面內的 ▶ ⏸ ↺ 按鈕控制，並即時調整速度。")

# ======================================================================================================
# 分頁 3：步數分佈演進
# ======================================================================================================
with tab_map["📈 步數分佈演進"]:
    st.subheader("🚶 隨機行走機率演進分析")

    # 公式說明：n 步轉移機率的遞迴展開式（對應圖片）
    st.markdown(f"""
    <div class="glass-card">
      <div class="calc-box">
        <strong style="font-size:1.1rem;">📏 數學依據：n 步轉移機率計算</strong>
        <div style="margin:10px 0;">
          $$(P^n)_{{ij}} = \\sum_{{k=1}}^{{m}} P_{{ik}}^{{(n-1)}} P_{{kj}}$$
        </div>
        <div class="explain-box">
          <strong>💡 邏輯解析：</strong><br>
          矩陣冪的遞迴展開式：$n$ 步轉移機率 = 先走 $n-1$ 步到中間節點 $k$，
          再從 $k$ 一步到達 $j$，對所有中間節點 $k$ 求和。
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        start_pos_evo = st.number_input("選擇出發位置 (Starting Node)", 1, n_nodes, 1)
    with col_s2:
        hours = st.number_input("設定時間 (小時/步數)", 0, 500, 1)

    # 計算 n 步後的分佈：v^(n) = v^(0) · P^n
    v0 = np.zeros(n_nodes)
    v0[start_pos_evo - 1] = 1.0
    Pm = np.linalg.matrix_power(P, hours)
    vm = np.dot(v0, Pm)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    res_c1, res_c2 = st.columns(2)
    with res_c1:
        st.markdown(f"**⏱️ 第 {hours} 步分佈表（從 {label_prefix} {start_pos_evo} 出發）**")
        df_vm = pd.DataFrame({
            "位置": [f"{label_prefix} {i+1}" for i in range(n_nodes)],
            "機率": vm
        })
        st.table(df_vm.style.format({"機率": "{:.4%}"}))
    with res_c2:
        fig_vm, ax_vm = plt.subplots(figsize=(5, 4))
        ax_vm.bar(df_vm["位置"], df_vm["機率"], color="#007AFF")
        ax_vm.set_ylim(0, 1.0)
        ax_vm.set_title(f"Distribution after {hours} step(s)")
        st.pyplot(fig_vm); plt.close(fig_vm)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("⏳ 長期演進趨勢圖")
    max_trace = st.slider("分析總時長 (步數)", 1, 100, 20)
    trace_data = []
    curr_v = v0.copy()
    for _ in range(max_trace + 1):
        trace_data.append(curr_v.copy())
        curr_v = np.dot(curr_v, P)

    df_trace = pd.DataFrame(
        trace_data, columns=[f"{label_prefix} {i+1}" for i in range(n_nodes)]
    )
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    fig_trace, ax_trace = plt.subplots(figsize=(10, 4))
    for col in df_trace.columns:
        ax_trace.plot(df_trace.index, df_trace[col], label=col, marker='.', markersize=4, alpha=0.8)
    ax_trace.set_title(f"Evolution from {label_prefix} {start_pos_evo}")
    ax_trace.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    st.pyplot(fig_trace); plt.close(fig_trace)
    st.markdown('</div>', unsafe_allow_html=True)

# ======================================================================================================
# 分頁 4：轉移矩陣
# ======================================================================================================
with tab_map["📊 轉移矩陣"]:
    st.subheader("轉移矩陣 $P$（每列加總 = 1）")
    st.caption("P[i][j] = 從節點 i 移動到節點 j 的機率。每一列的所有元素之和等於 1（機率守恆）。")
    df_P = pd.DataFrame(
        P,
        index=[f"{label_prefix} {i+1}" for i in range(n_nodes)],
        columns=[f"{label_prefix} {i+1}" for i in range(n_nodes)]
    )
    df_P['行加總 (Sum)'] = P.sum(axis=1)
    st.dataframe(df_P.style.format("{:.4f}"))

# ======================================================================================================
# 分頁 5：收斂趨勢
# ======================================================================================================
with tab_map["📉 收斂趨勢"]:
    st.subheader("收斂過程分析")
    st.caption("對數座標下，誤差曲線顯示每次迭代後分佈向穩態靠近的速度。")
    u_iters = st.slider("調整迭代次數 (Iterations)", 1, 500, 100)
    err_hist = get_convergence_history_fixed(P, u_iters)
    if err_hist:
        fig_conv, ax_conv = plt.subplots(figsize=(8, 4))
        ax_conv.plot(err_hist, color='#007AFF', lw=2, marker='o', markersize=2)
        ax_conv.set_yscale('log')
        ax_conv.set_xlabel("Number of iterations")
        ax_conv.set_ylabel("Maximum error (log scale)")
        ax_conv.grid(True, alpha=0.3)
        ax_conv.set_title("Convergence Error per Iteration")
        st.pyplot(fig_conv); plt.close(fig_conv)

# ======================================================================================================
# 分頁 6：穩定狀態 + 回傳時間（合併）
# ======================================================================================================
with tab_map["🎯 穩定狀態與回傳時間"]:
    st.subheader("長期分佈（穩態）與各格平均回傳時間")

    # 回傳時間定理說明
    st.markdown("""
    <div class="calc-box">
      <strong>📐 平均回傳時間定理（Mean First Return Time Theorem）</strong><br><br>
      對於不可約、正常返馬可夫鏈，從狀態 $i$ 出發，<strong>第一次回到 $i$ 的期望步數</strong>為：
      $$\\mu_i = \\frac{1}{\\pi_i}$$
      其中 $\\pi_i$ 為穩態機率。$\\pi_i$ 越大（越常被拜訪），回傳時間越短。
    </div>
    """, unsafe_allow_html=True)

    # 彙整表格：穩態機率 + 理論回傳時間
    rt_display = []
    for i in range(n_nodes):
        pi  = steady_v[i] if len(steady_v) > i else 0.0
        rt  = return_times[i] if len(return_times) > i else float('inf')
        rt_display.append({
            "節點":           f"{label_prefix} {i+1}",
            "穩態機率 π":     pi,
            "理論回傳步數 μ = 1/π": rt if not np.isinf(rt) else float('nan'),
        })
    df_rt = pd.DataFrame(rt_display)

    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.dataframe(df_rt.style.format({
            "穩態機率 π":         "{:.4%}",
            "理論回傳步數 μ = 1/π": "{:.2f}",
        }))
    with col_b:
        # 雙軸長條圖：π（左軸）與 μ（右軸）
        fig_rt, ax1 = plt.subplots(figsize=(5, 4))
        x     = np.arange(n_nodes)
        width = 0.4
        bars1 = ax1.bar(x - width/2, df_rt["穩態機率 π"], width, color="#007AFF", label="π（穩態）")
        ax1.set_ylabel("穩態機率 π", color="#007AFF")
        ax1.tick_params(axis='y', labelcolor="#007AFF")
        ax2 = ax1.twinx()
        valid_rt = df_rt["理論回傳步數 μ = 1/π"].fillna(0)
        bars2 = ax2.bar(x + width/2, valid_rt, width, color="#FF9500", alpha=0.8, label="μ = 1/π")
        ax2.set_ylabel("平均回傳步數 μ", color="#FF9500")
        ax2.tick_params(axis='y', labelcolor="#FF9500")
        ax1.set_xticks(x)
        ax1.set_xticklabels([f"{label_prefix}{i+1}" for i in range(n_nodes)], rotation=45)
        ax1.set_title("穩態機率 π vs 平均回傳步數 μ")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)
        plt.tight_layout()
        st.pyplot(fig_rt); plt.close(fig_rt)

    # 排行：最快 / 最慢回傳
    if len(return_times) > 0 and not all(np.isinf(return_times)):
        finite_rt = [(i+1, rt) for i, rt in enumerate(return_times) if not np.isinf(rt)]
        if finite_rt:
            fastest = min(finite_rt, key=lambda x: x[1])
            slowest = max(finite_rt, key=lambda x: x[1])
            fa_col, sl_col = st.columns(2)
            fa_col.metric("⚡ 最快回傳節點",
                          f"{label_prefix} {fastest[0]}",
                          f"μ = {fastest[1]:.2f} 步")
            sl_col.metric("🐢 最慢回傳節點",
                          f"{label_prefix} {slowest[0]}",
                          f"μ = {slowest[1]:.2f} 步")

# ======================================================================================================
# 分頁 7：計算詳情
# ======================================================================================================
with tab_map["📝 計算詳情"]:
    st.subheader("🔍 數值計算過程解剖")
    calc_mode = st.selectbox(
        "選擇計算類型",
        ["轉移矩陣元素 $P_{ij}$", "穩定狀態元素 $\\pi_i$",
         "矩陣乘法 $(P^2)_{ij}$", "回傳時間 $\\mu_i$"]
    )

    if calc_mode == "轉移矩陣元素 $P_{ij}$":
        c1, c2 = st.columns(2)
        with c1: row = st.number_input("選擇行（起點 $i$）", 1, n_nodes, 1)
        with c2: col = st.number_input("選擇列（終點 $j$）", 1, n_nodes, min(2, n_nodes))
        weight_ij = next((w for v, w in adj[row] if v == col), 0.0)
        self_w  = 1.0 if allow_self else 0.0
        total_w = sum(w for _, w in adj[row]) + self_w
        res     = P[row-1, col-1]
        st.markdown('<div class="calc-box">', unsafe_allow_html=True)
        st.latex(rf"P_{{{row},{col}}} = \frac{{w_{{{row} \to {col}}}}}{{\sum_{{\text{{neighbors}}}} w + w_{{\text{{self}}}}}}")
        st.latex(rf"P_{{{row},{col}}} = \frac{{{weight_ij:.1f}}}{{{total_w - self_w:.1f} + {self_w:.1f}}} = {res:.4f}")
        st.markdown(f'<div class="explain-box"><strong>💡 邏輯：</strong>'
                    f'從{label_prefix} {row} 出發，移動到{label_prefix} {col} 的邊權重佔總權重的比例。</div>',
                    unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    elif calc_mode == "穩定狀態元素 $\\pi_i$":
        node = st.number_input("選擇位置 $i$", 1, n_nodes, 1)
        formula_terms = [f"{steady_v[j-1]:.4f} \\times {P[j-1, node-1]:.4f}"
                         for j in range(1, n_nodes + 1)]
        result = sum(steady_v[j-1] * P[j-1, node-1] for j in range(1, n_nodes + 1))
        st.markdown('<div class="calc-box">', unsafe_allow_html=True)
        st.latex(rf"\pi_{{{node}}} = \sum_{{j=1}}^{{{n_nodes}}} (\pi_{{j}} \times P_{{j,{node}}})")
        st.latex(rf"\pi_{{{node}}} = {' + '.join(formula_terms)} = {result:.4f}")
        st.markdown(f'<div class="explain-box"><strong>💡 邏輯：</strong>'
                    f'長期處於{label_prefix} {node} 的機率，等於所有能到達此處的節點貢獻之總和。</div>',
                    unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    elif calc_mode == "矩陣乘法 $(P^2)_{ij}$":
        c1, c2 = st.columns(2)
        with c1: r = st.number_input("選擇行 $i$", 1, n_nodes, 1)
        with c2: c = st.number_input("選擇列 $j$", 1, n_nodes, 1)
        formula_terms = [f"{P[r-1, k-1]:.2f} \\times {P[k-1, c-1]:.2f}"
                         for k in range(1, n_nodes + 1)]
        result = sum(P[r-1, k-1] * P[k-1, c-1] for k in range(1, n_nodes + 1))
        st.markdown('<div class="calc-box">', unsafe_allow_html=True)
        st.latex(rf"(P^2)_{{r,c}} = \sum_{{k=1}}^{{n_{{nodes}}}} (P_{{r,k}} \times P_{{k,c}})")
        st.latex(rf"(P^2)_{{{r},{c}}} = {' + '.join(formula_terms)} = {result:.4f}")
        st.markdown(f'<div class="explain-box"><strong>💡 邏輯：</strong>'
                    f'經過恰好 2 步，從{label_prefix} {r} 到達{label_prefix} {c} 的機率。</div>',
                    unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    elif calc_mode == "回傳時間 $\\mu_i$":
        # 展示回傳時間的計算過程
        node = st.number_input("選擇節點 $i$", 1, n_nodes, 1)
        pi_i = steady_v[node-1] if len(steady_v) >= node else 0.0
        mu_i = return_times[node-1] if len(return_times) >= node else float('inf')
        st.markdown('<div class="calc-box">', unsafe_allow_html=True)
        st.latex(rf"\mu_{{{node}}} = \frac{{1}}{{\pi_{{{node}}}}}")
        if not np.isinf(mu_i):
            st.latex(rf"\mu_{{{node}}} = \frac{{1}}{{{pi_i:.6f}}} = {mu_i:.4f} \text{{ 步}}")
        else:
            st.latex(rf"\mu_{{{node}}} = \infty \quad (\pi_{{{node}}} = 0，不可達狀態)")
        st.markdown(f'<div class="explain-box"><strong>💡 邏輯：</strong>'
                    f'從{label_prefix} {node} 出發，平均需要 <strong>{mu_i:.2f} 步</strong> 才會再次回到此處。'
                    f'穩態機率越高（π = {pi_i:.4%}），回傳速度越快。</div>',
                    unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ======================================================================================================
# 分頁 8：數學原理
# ======================================================================================================
with tab_map["📐 數學原理"]:
    st.subheader("📐 數學模型與解析")
    if mode == "🐁 8格迷宮老鼠 (Mouse Maze)":
        st.markdown("""
        ### 迷宮問題分析
        - **無自環限制**：$P_{ii} = 0$，老鼠每步必須移動到相鄰格子。
        - **邊界節點**（位置 1 與 8）：只有一個鄰居，因此每步移動機率為 1。
        - **多步轉移**：使用矩陣冪 $P^m$ 求 $m$ 步後的分佈。
        - **穩態**：線性鏈的穩態分佈與度數成正比（中間節點各有 2 個鄰居）。
        - **回傳時間**：邊界節點（度數=1）穩態機率最低，回傳時間最長。
        """)
    else:
        st.markdown("""
        ### 巡邏問題分析
        - **自環權重**：$w_{ii} = 1.0$，警察可以停留在原路口（自環）。
        - **轉移機率**：$P_{ij} = w_{ij} / (\\sum_{k} w_{ik} + 1)$，分母包含自環。
        - **穩態解讀**：機率越高的路口，長期被巡邏的頻率越高。
        - **回傳時間**：穩態機率高的路口回傳快（$\\mu_i = 1/\\pi_i$）。
        """)

    st.markdown("""
    ### 共通定理
    | 定理 | 公式 | 說明 |
    |------|------|------|
    | 穩態方程 | $\\pi = \\pi P$ | 穩態不因轉移而改變 |
    | 機率守恆 | $\\sum_i \\pi_i = 1$ | 所有節點機率之和為 1 |
    | 平均回傳時間 | $\\mu_i = 1/\\pi_i$ | 與穩態機率成反比 |
    | 細緻平衡（若成立） | $\\pi_i P_{ij} = \\pi_j P_{ji}$ | 可逆鏈的充要條件 |
    """)

# ======================================================================================================
# 分頁 9（迷宮專用）：矩陣運算分析
# ======================================================================================================
if "🧮 矩陣運算分析" in tab_map:
    with tab_map["🧮 矩陣運算分析"]:
        st.subheader("🎯 特定步數機率計算")
        col_input, col_res = st.columns([1, 2])
        with col_input:
            start_node_m  = st.number_input("設定起始位置 $v^{(0)}$", 1, n_nodes, 1)
            target_node_m = st.number_input("設定目標位置", 1, n_nodes, min(5, n_nodes))
            steps_m       = st.number_input("計算步數 $m$", 1, 100, 2)
        with col_res:
            Pm_m  = np.linalg.matrix_power(P, steps_m)
            v0_m  = np.zeros(n_nodes)
            v0_m[start_node_m - 1] = 1.0
            vm_m  = np.dot(v0_m, Pm_m)
            st.metric(
                f"經過 {steps_m} 步後，在位置 {target_node_m} 的機率",
                f"{vm_m[target_node_m-1]:.4%}"
            )
            df_vm_m = pd.DataFrame({"位置": range(1, n_nodes+1), "機率": vm_m})
            st.bar_chart(df_vm_m.set_index("位置")["機率"])

            # 附帶顯示此目標節點的理論回傳時間
            pi_t  = steady_v[target_node_m - 1]
            mu_t  = return_times[target_node_m - 1]
            st.markdown(f"""
            <div class="explain-box">
            <strong>節點 {target_node_m} 的關鍵資訊：</strong><br>
            穩態機率 π = {pi_t:.4%}　｜　
            理論回傳時間 μ = {mu_t:.2f} 步
            </div>
            """, unsafe_allow_html=True)
