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
    P = np.zeros((n, n))
    adj = {i: [] for i in range(1, n + 1)}
    for u, v, w in edges_with_weights:
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
    v = np.ones(n) / n 
    error_history = []
    iteration = 0
    while True:
        v_next = np.dot(v, P)
        error = np.max(np.abs(v_next - v))
        error_history.append(error)
        if error < threshold or iteration > 10000:
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
        probs = probs / np.sum(probs)
        current = np.random.choice(range(1, n + 1), p=probs)
        path.append(current)
    return path

# ================== 視覺化模組 ==================

def create_interactive_graph(n, edges_with_weights, steady_v=None):
    net = Network(height="500px", width="100%", bgcolor="#ffffff", font_color="black")
    net.barnes_hut()
    for i in range(1, n + 1):
        color = "#ADD8E6" 
        if steady_v is not None:
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

st.set_page_config(page_title="馬可夫鏈分析工作站", layout="wide")
st.title("👮 交通警察值勤分析工作站 (連動版)")

# --- 側邊欄：連動設定 ---
st.sidebar.header("⚙️ 系統配置")

# 1. 選擇佈局
layout_type = st.sidebar.selectbox("連接佈局", ["3x4 網格", "自定義網格", "手動輸入 (u,v,w)"])

# 用於儲存最終確定的 n_nodes 和 edges
n_nodes = 12
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
    # --- 連動核心：自動偵測 vs 手動指定 ---
    node_mode = st.sidebar.radio("路口總數設定方式", ["自動偵測 (根據輸入內容)", "手動指定 (限制範圍)"])
    
    st.sidebar.info("格式：`路口1,路口2,權重` (每行一組)")
    raw_input = st.sidebar.text_area("輸入關係", "1,2,1.0\n2,3,2.0\n3,4,1.0\n4,1,1.0\n1,5,0.5")
    
    # 暫時解析輸入內容
    temp_edges = []
    max_node_found = 0
    for line in raw_input.split('\n'):
        if line.strip():
            try:
                u, v, w = map(float, line.split(','))
                temp_edges.append((int(u), int(v), w))
                max_node_found = max(max_node_found, int(u), int(v))
            except:
                st.sidebar.error(f"❌ 格式錯誤: {line}")

    if node_mode == "自動偵測 (根據輸入內容)":
        n_nodes = max_node_found if max_node_found > 0 else 2
        st.sidebar.success(f"🔍 已自動偵測路口總數為: {n_nodes}")
        edges_with_weights = temp_edges
    else:
        manual_n = st.sidebar.number_input("請輸入路口總數", min_value=2, value=12)
        n_nodes = manual_n
        # 過濾掉超出範圍的邊
        for u, v, w in temp_edges:
            if u <= n_nodes and v <= n_nodes:
                edges_with_weights.append((u, v, w))
            else:
                st.sidebar.warning(f"⚠️ 邊 {u}-{v} 已被忽略 (超出總數 {n_nodes})")

threshold = st.sidebar.number_input("收斂閾值", value=0.000001, format="%.7f")

# --- 計算區 ---
# 確保在沒有輸入任何邊時，n_nodes 至少為 2 以免矩陣出錯
if n_nodes < 2: n_nodes = 2

P = build_transition_matrix(n_nodes, edges_with_weights)
steady_v, iters, error_hist = find_steady_state(P, threshold)

# --- 介面分頁 ---
tab_graph, tab_sim, tab_matrix, tab_conv, tab_steady = st.tabs([
    "🌐 互動拓撲圖", "⏱️ 隨機行走模擬", "📊 轉移矩陣", "📉 收斂分析", "🎯 穩定狀態"
])

with tab_graph:
    st.subheader("互動式路口連接圖")
    st.info(f"目前系統設定路口總數: {n_nodes}")
    graph_html = create_interactive_graph(n_nodes, edges_with_weights, steady_v)
    with open(graph_html, 'r', encoding='utf-8') as f:
        components.html(f.read(), height=550)

with tab_sim:
    st.subheader("警察隨機行走模擬")
    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1: start_node = st.number_input("起始路口", 1, n_nodes, 1)
    with col_s2: sim_steps = st.slider("模擬時間 (小時)", 1, 100, 20)
    with col_s3: speed = st.slider("動畫速度", 0.1, 1.0, 0.3)
    if st.button("開始模擬行走"):
        path = run_simulation(P, start_node, sim_steps)
        st.write(f"🚶 模擬路徑: {' $\rightarrow$ '.join(map(str, path))}")
        progress_bar = st.progress(0)
        status_text = st.empty()
        for i in range(len(path)):
            status_text.text(f"目前位置: 路口 {path[i]}")
            progress_bar.progress((i + 1) / len(path))
            time.sleep(speed)
        st.success("模擬完成！")

with tab_matrix:
    st.subheader("轉移矩陣 $P$")
    df_P = pd.DataFrame(P, index=[f"路口 {i+1}" for i in range(n_nodes)], 
                        columns=[f"路口 {i+1}" for i in range(n_nodes)])
    st.dataframe(df_P.style.format("{:.4f}"))
    st.download_button("📥 下載矩陣為 CSV", df_P.to_csv().encode('utf-8'), "matrix.csv", "text/csv")

with tab_conv:
    st.subheader("迭代收斂過程")
    st.write(f"迭代次數: {iters}")
    fig_conv, ax_conv = plt.subplots()
    ax_conv.plot(error_hist, color='blue', lw=2)
    ax_conv.set_yscale('log') 
    ax_conv.set_xlabel("迭代次數")
    ax_conv.set_ylabel("最大誤差 (Log)")
    ax_conv.grid(True, which="both", ls="-", alpha=0.5)
    st.pyplot(fig_conv)

with tab_steady:
    st.subheader("長期值勤分佈")
    df_steady = pd.DataFrame({"路口": [f"路口 {i+1}" for i in range(n_nodes)], "機率": steady_v})
    st.table(df_steady.style.format({"機率": "{:.4%}"}))
    st.bar_chart(df_steady.set_index("路口")["機率"])
