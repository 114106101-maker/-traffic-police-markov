import streamlit as st
import numpy as np
import pandas as pd

# ----------------- 1. 定義路口結構 (假設 3x4 網格) -----------------
def create_transition_matrix():
    # 定義 3x4 網格的鄰居關係
    # 節點編號 1-12
    adj = {
        1: [2, 5], 2: [1, 3, 6], 3: [2, 4, 7], 4: [3, 8],
        5: [1, 6, 9], 6: [2, 5, 7, 10], 7: [3, 6, 8, 11], 8: [4, 7, 12],
        9: [5, 10], 10: [6, 9, 11], 11: [7, 10, 12], 12: [8, 11]
    }
    
    n = 12
    P = np.zeros((n, n))
    
    for i in range(1, n + 1):
        neighbors = adj[i]
        prob = 1.0 / (len(neighbors) + 1) # 包含自己
        # 停留原處
        P[i-1, i-1] = prob
        # 移動到鄰居
        for nb in neighbors:
            P[i-1, nb-1] = prob
            
    return P

# ----------------- 2. 計算函數 -----------------
def get_prob_after_steps(P, start_node, end_node, steps):
    # 起始向量 v0
    v0 = np.zeros(12)
    v0[start_node - 1] = 1.0
    # 計算 P^n
    Pn = np.linalg.matrix_power(P, steps)
    # 結果向量 vn = v0 * Pn
    vn = np.dot(v0, Pn)
    return vn[end_node - 1]

def find_steady_state(P):
    n = P.shape[0]
    v = np.ones(n) / n # 初始均分
    threshold = 0.000001
    iteration = 0
    
    while True:
        v_next = np.dot(v, P)
        if np.max(np.abs(v_next - v)) < threshold:
            break
        v = v_next
        iteration += 1
    return v, iteration

# ----------------- 3. Streamlit 介面 -----------------
st.set_page_config(page_title="交通警察值勤分析系統", layout="wide")

st.title("👮 交通警察 12 個路口值勤分配分析")
st.markdown("""
本程式根據馬可夫鏈 (Markov Chain) 計算警察在 12 個路口的轉移機率與穩定狀態。
**假設條件**：路口採 $3 \times 4$ 網格佈局，停留原處與移動至鄰近路口的機率相等。
""")

P = create_transition_matrix()

# (a) 轉移矩陣
st.subheader("(a) 轉移矩陣 (Transition Matrix)")
df_P = pd.DataFrame(P, index=[f"路口 {i+1}" for i in range(12)], 
                    columns=[f"路口 {i+1}" for i in range(12)])
st.dataframe(df_P.style.format("{:.4f}"))

# (b) & (c) 特定路徑機率
st.subheader("(b) & (c) 特定時間後到達機率")
col1, col2, col3 = st.columns(3)
with col1:
    start_node = st.number_input("起始位置", min_value=1, max_value=12, value=1)
with col2:
    end_node = st.number_input("目標位置", min_value=1, max_value=12, value=5)
with col3:
    steps = st.selectbox("時間 (小時)", [2, 10])

prob = get_prob_after_steps(P, start_node, end_node, steps)
st.success(f"從路口 {start_node} 出發，經過 {steps} 小時後在路口 {end_node} 的機率為: **{prob:.6f}**")

# (d) & (e) 穩定狀態
st.subheader("(d) & (e) 穩定狀態分析 (Steady State)")
steady_v, iters = find_steady_state(P)
st.write(f"系統在經過 {iters} 次迭代後達到穩定狀態 (誤差 < 0.000001)")

df_steady = pd.DataFrame({
    "路口": [f"路口 {i+1}" for i in range(12)],
    "值勤時間比例": steady_v
})
st.table(df_steady.style.format({"值勤時間比例": "{:.4%}"}))

st.info("💡 **數學提示**：穩定狀態代表長期來看，警察出現在各路口的機率分佈。在無方向性圖中，穩定機率通常與該節點的度數 (Degree) 成正比。")