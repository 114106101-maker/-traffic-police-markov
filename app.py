import streamlit as st
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# ----------------- 1. 核心計算邏輯 -----------------

def build_transition_matrix(n, edges):
    P = np.zeros((n, n))
    adj = {i: [] for i in range(1, n + 1)}
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    for i in range(1, n + 1):
        neighbors = adj[i]
        prob = 1.0 / (len(neighbors) + 1)
        P[i-1, i-1] = prob
        for nb in neighbors:
            P[i-1, nb-1] = prob
    return P

def get_prob_after_steps(P, start_node, end_node, steps):
    n = P.shape[0]
    v0 = np.zeros(n)
    v0[start_node - 1] = 1.0
    Pn = np.linalg.matrix_power(P, steps)
    vn = np.dot(v0, Pn)
    return vn[end_node - 1]

def find_steady_state(P, threshold):
    n = P.shape[0]
    v = np.ones(n) / n 
    iteration = 0
    while True:
        v_next = np.dot(v, P)
        if np.max(np.abs(v_next - v)) < threshold:
            break
        v = v_next
        iteration += 1
        if iteration > 10000: break
    return v, iteration

# ----------------- 2. 視覺化繪圖函數 -----------------

def draw_graph(n, edges, steady_v=None):
    # 建立 NetworkX 圖形
    G = nx.Graph()
    G.add_nodes_from(range(1, n + 1))
    G.add_edges_from(edges)
    
    plt.figure(figsize=(8, 6))
    
    # 嘗試使用 spring_layout 使圖形分佈均勻
    pos = nx.spring_layout(G, seed=42) 
    
    # 如果有穩定狀態向量，則用顏色表示機率
    if steady_v is not None:
        node_color = steady_v
        cmap = plt.cm.Blues # 使用藍色系
        colorbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap), ax=plt.gca())
        colorbar.set_label('穩定狀態機率 (Duty Probability)')
    else:
        node_color = 'skyblue'

    # 繪製節點
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color=node_color, 
                           cmap=plt.cm.Blues, edgecolors='grey')
    # 繪製邊線
    nx.draw_networkx_edges(G, pos, width=2, edge_color='gray', alpha=0.6)
    # 繪製標籤
    nx.draw_networkx_labels(G, pos, font_size=12, font_family='sans-serif')
    
    plt.title("路口連接拓撲圖", fontsize=15)
    plt.axis('off')
    return plt.gcf()

# ----------------- 3. Streamlit 介面 -----------------

st.set_page_config(page_title="通用馬可夫鏈分析器", layout="wide")

st.title("👮 交通警察值勤分析系統 (視覺化版)")
st.markdown("您可以自定義路口數量、連接方式，並直接觀察路口拓撲圖。")

# --- 側邊欄 ---
st.sidebar.header("⚙️ 參數設定")
n_nodes = st.sidebar.number_input("路口總數 (n)", min_value=2, max_value=50, value=12)

layout_type = st.sidebar.selectbox("選擇路口連接佈局", ["3x4 網格 (預設)", "自定義網格 (R x C)", "手動輸入連接清單"])

edges = []
if layout_type == "3x4 網格 (預設)":
    for r in range(3):
        for c in range(4):
            u = r * 4 + c + 1
            if c < 3: edges.append((u, u + 1))
            if r < 2: edges.append((u, u + 4))
elif layout_type == "自定義網格 (R x C)":
    rows = st.sidebar.number_input("行數 (Rows)", min_value=1, max_value=10, value=3)
    cols = st.sidebar.number_input("列數 (Cols)", min_value=1, max_value=10, value=4)
    if rows * cols == n_nodes:
        for r in range(rows):
            for c in range(cols):
                u = r * cols + c + 1
                if c < cols - 1: edges.append((u, u + 1))
                if r < rows - 1: edges.append((u, u + cols))
    else:
        st.sidebar.error(f"⚠️ {rows}x{cols} 不等於 {n_nodes}")
elif layout_type == "手動輸入連接清單":
    raw_input = st.sidebar.text_area("輸入連接關係 (u,v)", "1,2\n2,3\n3,4\n4,1\n1,5")
    for line in raw_input.split('\n'):
        if line.strip():
            try:
                u, v = map(int, line.split(','))
                if 1 <= u <= n_nodes and 1 <= v <= n_nodes: edges.append((u, v))
            except: pass

threshold = st.sidebar.number_input("穩定狀態閾值", value=0.000001, format="%.7f")

# --- 計算 ---
P = build_transition_matrix(n_nodes, edges)
steady_v, iters = find_steady_state(P, threshold)

# --- 介面分頁 ---
tab_graph, tab_matrix, tab_prob, tab_steady = st.tabs(["🌐 連接圖", "📊 轉移矩陣", "⏱️ 機率預測", "🎯 穩定狀態"])

with tab_graph:
    st.subheader("路口連接視覺化")
    st.info("提示：節點顏色越深，代表長期值勤機率越高。")
    fig = draw_graph(n_nodes, edges, steady_v)
    st.pyplot(fig)

with tab_matrix:
    st.subheader("轉移矩陣 $P$")
    df_P = pd.DataFrame(P, index=[f"路口 {i+1}" for i in range(n_nodes)], 
                        columns=[f"路口 {i+1}" for i in range(n_nodes)])
    st.dataframe(df_P.style.format("{:.4f}"))

with tab_prob:
    st.subheader("特定時間後的機率")
    c1, c2, c3 = st.columns(3)
    with c1: start_n = st.number_input("起始位置", min_value=1, max_value=n_nodes, value=1)
    with c2: end_n = st.number_input("目標位置", min_value=1, max_value=n_nodes, value=5 if n_nodes >= 5 else 2)
    with c3: steps = st.number_input("時間 (小時)", min_value=1, max_value=1000, value=10)
    if st.button("計算機率"):
        prob = get_prob_after_steps(P, start_n, end_n, steps)
        st.success(f"從路口 {start_n} 出發，經過 {steps} 小時後在路口 {end_n} 的機率為: **{prob:.6f}**")

with tab_steady:
    st.subheader("長期值勤分佈 (穩定狀態)")
    st.write(f"迭代次數: {iters}")
    df_steady = pd.DataFrame({"路口": [f"路口 {i+1}" for i in range(n_nodes)], "長期值勤比例": steady_v})
    st.table(df_steady.style.format({"長期值勤比例": "{:.4%}"}))
    st.bar_chart(df_steady.set_index("路口")["長期值勤比例"])
