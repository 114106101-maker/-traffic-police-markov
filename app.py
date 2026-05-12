import streamlit as st
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
import streamlit.components.v1 as components
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

# ================== 視覺化與動畫模組 ==================

def create_interactive_graph(n, edges_with_weights, steady_v=None):
    net = Network(height="500px", width="100%", bgcolor="#ffffff", font_color="black")
    net.barnes_hut()
    for i in range(1, n + 1):
        color = "#ADD8E6" 
        if steady_v is not None and len(steady_v) >= i:
            intensity = int(steady_v[i-1] * 255 * 2) 
            color = f"rgb(255, {255-min(intensity, 255)}, {255-min(intensity, 255)})"
        net.add_node(i, label=f"路口 {i}", color=color)
    for u, v, w in edges_with_weights:
        if 1 <= u <= n and 1 <= v <= n:
            net.add_edge(u, v, value=w)
    net.save_graph("graph.html")
    return "graph.html"

def draw_simulation_frame(n, edges, current_node, steady_v):
    """繪製模擬中的單一幀圖形"""
    G = nx.Graph()
    G.add_nodes_from(range(1, n + 1))
    G.add_edges_from([(u, v) for u, v, w in edges])
    
    # 固定佈局，防止動畫跳動
    pos = nx.spring_layout(G, seed=42)
    
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # 節點顏色：目前位置為亮黃色，其餘根據穩定機率
    node_colors = []
    for i in range(1, n + 1):
        if i == current_node:
            node_colors.append("#FFFF00") # 亮黃色 (警察位置)
        elif steady_v is not None:
            # 根據機率給予淺藍色
            node_colors.append("#ADD8E6")
        else:
            node_colors.append("#ADD8E6")

    nx.draw(G, pos, with_labels=True, node_color=node_colors, 
            node_size=500, edge_color="#CCCCCC", font_size=10, ax=ax)
    ax.set_title(f"警察目前位置: 路口 {current_node}", fontsize=14)
    plt.axis('off')
    return fig

# ================== Streamlit 介面 ==================

st.set_page_config(page_title="馬可夫分析站 Pro", layout="wide")
st.title("👮 交通警察值勤分析工作站 (動畫地圖版)")

# --- 側邊欄 ---
st.sidebar.header("⚙️ 配置中心")
layout_type = st.sidebar.selectbox("連接佈局", ["3x4 網格", "自定義網格", "手動輸入 (u,v,w)"])

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
    raw_input = st.sidebar.text_area("編輯關係 (u,v,w)", "1,2,1.0\n2,3,1.0\n3,1,1.0")
    temp_edges = []
    current_max = 0
    for line in raw_input.split('\n'):
        if line.strip():
            try:
                u, v, w = map(float, line.split(','))
                temp_edges.append((int(u), int(v), w))
                current_max = max(current_max, int(u), int(v))
            except: pass
    n_nodes = current_max if current_max >= 2 else 2
    edges_with_weights = temp_edges

threshold = st.sidebar.number_input("收斂閾值", value=0.000001, format="%.7f")

# --- 計算 ---
P = build_transition_matrix(n_nodes, edges_with_weights)
steady_v, iters, error_hist = find_steady_state(P, threshold)

# --- 分頁 ---
tab_graph, tab_sim, tab_matrix, tab_conv, tab_steady = st.tabs([
    "🌐 互動拓撲圖", "⏱️ 隨機行走模擬", "📊 轉移矩陣", "📉 收 converge 分析", "🎯 穩定狀態"
])

with tab_graph:
    st.subheader("互動式路口連接圖")
    graph_html = create_interactive_graph(n_nodes, edges_with_weights, steady_v)
    with open(graph_html, 'r', encoding='utf-8') as f:
        components.html(f.read(), height=550)

with tab_sim:
    st.subheader("警察隨機行走模擬 (動畫地圖)")
    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1: start_node = st.number_input("起始路口", 1, n_nodes, 1)
    with col_s2: sim_steps = st.slider("模擬時間", 1, 100, 20)
    with col_s3: speed = st.slider("動畫速度", 0.1, 1.0, 0.3)
    
    if st.button("🚀 開始模擬行走"):
        # 建立一個容器來存放動畫地圖
        map_placeholder = st.empty()
        status_placeholder = st.empty()
        
        current = start_node
        path = [current]
        
        for i in range(sim_steps):
            # 1. 更新地圖視覺化
            fig = draw_simulation_frame(n_nodes, edges_with_weights, current, steady_v)
            map_placeholder.pyplot(fig)
            
            # 2. 更新狀態文字
            status_placeholder.info(f"⏰ 第 {i+1} 小時: 警察目前位於 **路口 {current}**")
            
            # 3. 計算下一個位置
            probs = P[current-1, :]
            probs = probs / np.sum(probs)
            current = np.random.choice(range(1, n_nodes + 1), p=probs)
            path.append(current)
            
            # 4. 暫停以產生動畫效果
            time.sleep(speed)
            plt.close(fig) # 釋放記憶體
            
        st.success(f"模擬結束！完整路徑: {' $\rightarrow$ '.join(map(str, path))}")

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
        st.pyplot(fig_conv)

with tab_steady:
    st.subheader("長期值勤分佈")
    df_steady = pd.DataFrame({"路口": [f"路口 {i+1}" for i in range(n_nodes)], "機率": steady_v})
    st.table(df_steady.style.format({"機率": "{:.4%}"}))
    st.bar_chart(df_steady.set_index("路口")["機率"])
