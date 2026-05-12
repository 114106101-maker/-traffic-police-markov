import streamlit as st
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
import streamlit.components.v1 as components
import random
import time

# ================== 核心計算模組 ==================

def build_transition_matrix(n, edges_with_weights):
    """根據動態的 n 建立轉移矩陣"""
    P = np.zeros((n, n))
    adj = {i: [] for i in range(1, n + 1)}
    for u, v, w in edges_with_weights:
        # 確保邊在目前動態範圍內
        if 1 <= u <= n and 1 <= v <= n:
            adj[u].append((v, w))
            adj[v].append((u, w))
    
    for i in range(1, n + 1):
        neighbors = adj[i]
        self_weight = 1.0 
        total_weight = sum([w for v, w in neighbors]) + self_weight
        P[i-1, i-1] = self_weight / total_weight
        for v, w in neighbors:
            P[i-1, v-1] = w / total_weight
    return P

def find_steady_state(P, threshold):
    n = P.shape[0]
    if n == 0: return np.array([]), 0, []
    v = np.ones(n) / n 
    error_history = []
    iteration = 0
    while True:
        v_next = np.dot(v, P)
        error = np.max(np.abs(v_next - v))
        error_history.append(error)
        if error < threshold or iteration > 5000:
            break
        v = v_next
        iteration += 1
    return v, iteration, error_history

def run_simulation(P, start_node, steps):
    n = P.shape[0]
    current = start_node
    path = [current]
    for _ in range(steps):
        probs = P[current-1, :]
        probs = probs / np.sum(probs) # 數值修正
        current = np.random.choice(range(1, n + 1), p=probs)
        path.append(current)
    return path

# ================== 視覺化模組 ==================

def create_interactive_graph(n, edges_with_weights, steady_v=None):
    net = Network(height="500px", width="100%", bgcolor="#ffffff", font_color="black")
    net.barnes_hut()
    for i in range(1, n + 1):
        color = "#ADD8E6" 
        if steady_v is not None and len(steady_v) >= i:
            intensity = int(steady_v[i-1] * 255 * 2) 
            color = f"rgb(255, {255-min(intensity, 255)}, {255-min(intensity, 255)})"
        net.add_node(i, label=f"路口 {i}", color=color, 
                     title=f"穩定機率: {steady_v[i-1]:.4f}" if steady_v is not None else "")
    for u, v, w in edges_with_weights:
        if 1 <= u <= n and 1 <= v <= n:
            net.add_edge(u, v, value=w, title=f"權重: {w}")
    net.save_graph("graph.html")
    return "graph.html"

# ================== Streamlit 介面 =================

st.set_page_config(page_title="動態馬可夫分析站", layout="wide")
st.title("👮 交通警察值勤分析工作站 (全動態連動版)")
st.markdown("💡 **功能**：路口總數將隨您的輸入**即時增加或減少**。")

# --- 側邊欄 ---
st.sidebar.header("⚙️ 配置中心")
layout_type = st.sidebar.selectbox("連接佈局", ["3x4 網格", "自定義網格", "手動輸入 (u,v,w)"])

# 初始化變數
n_nodes = 2
edges_with_weights = []

if layout_type == "3x4 網格":
    n_nodes = 12
    for r in range(3):
        for c in range(4):
            u = r * 4 + c + 1
            if c < 3: edges_with_weights.append((u, u + 1, 1.0))
            if r < 2: edges_with_weights.append((u, u + 4, 1.0))

elif layout_type == "自定義網格":
    rows = st.sidebar.number_input("行數", min_value=1, value=3)
    cols = st.sidebar.number_input("列數", min_value=1, value=4)
    n_nodes = rows * cols
    for r in range(rows):
        for c in range(cols):
            u = r * cols + c + 1
            if c < cols - 1: edges_with_weights.append((u, u + 1, 1.0))
            if r < rows - 1: edges_with_weights.append((u, u + cols, 1.0))

elif layout_type == "手動輸入 (u,v,w)":
    st.sidebar.info("輸入方式：`路口,路口,權重` (例如: 1,2,1.0)")
    raw_input = st.sidebar.text_area("編輯路口連接關係", "1,2,1.0\n2,3,1.0\n3,1,1.0")
    
    temp_edges = []
    current_max = 0
    for line in raw_input.split('\n'):
        if line.strip():
            try:
                parts = line.split(',')
                u, v, w = int(float(parts[0])), int(float(parts[1])), float(parts[2])
                temp_edges.append((u, v, w))
                current_max = max(current_max, u, v)
            except:
                st.sidebar.error(f"❌ 行格式錯誤: {line}")
    
    # 【關鍵連動】：路口總數 = 目前輸入中出現的最大編號
    n_nodes = current_max if current_max >= 2 else 2
    edges_with_weights = temp_edges
    st.sidebar.success(f"🎯 目前偵測到路口總數: {n_nodes}")

threshold = st.sidebar.number_input("收斂閾值", value=0.000001, format="%.7f")

# --- 計算區 ---
P = build_transition_matrix(n_nodes, edges_with_weights)
steady_v, iters, error_hist = find_steady_state(P, threshold)

# --- 分頁介面 ---
tab_graph, tab_sim, tab_matrix, tab_conv, tab_steady = st.tabs([
    "🌐 互動拓撲圖", "⏱️ 隨機行走模擬", "📊 轉移矩陣", "📉 收斂分析", "🎯 穩定狀態"
])

with tab_graph:
    st.subheader("動態路口連接圖")
    st.caption(f"系統目前運算規模: {n_nodes} 個路口")
    graph_html = create_interactive_graph(n_nodes, edges_with_weights, steady_v)
    with open(graph_html, 'r', encoding='utf-8') as f:
        components.html(f.read(), height=550)

with tab_sim:
    st.subheader("隨機行走模擬")
    # 起始路口隨 n_nodes 動態調整
    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1: start_node = st.number_input("起始路口", 1, n_nodes, 1)
    with col_s2: sim_steps = st.slider("模擬時間", 1, 100, 20)
    with col_s3: speed = st.slider("動畫速度", 0.1, 1.0, 0.3)
    
    if st.button("執行模擬"):
        path = run_simulation(P, start_node, sim_steps)
        st.write(f"🚶 路徑: {' $\rightarrow$ '.join(map(str, path))}")
        progress_bar = st.progress(0)
        status_text = st.empty()
        for i in range(len(path)):
            status_text.text(f"警察目前在: 路口 {path[i]}")
            progress_bar.progress((i + 1) / len(path))
            time.sleep(speed)
        st.success("模擬完成")

with tab_matrix:
    st.subheader("動態轉移矩陣")
    df_P = pd.DataFrame(P, index=[f"路口 {i+1}" for i in range(n_nodes)], 
                        columns=[f"路口 {i+1}" for i in range(n_nodes)])
    st.dataframe(df_P.style.format("{:.4f}"))

with tab_conv:
    st.subheader("收斂分析")
    if len(error_hist) > 0:
        fig_conv, ax_conv = plt.subplots()
        ax_conv.plot(error_hist, color='blue')
        ax_conv.set_yscale('log')
        ax_conv.set_xlabel("迭代次數")
        ax_conv.set_ylabel("誤差")
        st.pyplot(fig_conv)

with tab_steady:
    st.subheader("長期值勤分佈")
    df_steady = pd.DataFrame({"路口": [f"路口 {i+1}" for i in range(n_nodes)], "機率": steady_v})
    st.table(df_steady.style.format({"機率": "{:.4%}"}))
    st.bar_chart(df_steady.set_index("路口")["機率"])
