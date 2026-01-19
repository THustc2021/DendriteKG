import streamlit as st
import pandas as pd
import numpy as np

# 页面配置
st.set_page_config(
    page_title="简单 Streamlit App",
    page_icon="🚀",
    layout="centered"
)

# 标题
st.title("🚀 我的第一个 Streamlit App")
st.write("这是一个基于 Streamlit 的简单示例应用。")

# 文本输入
name = st.text_input("请输入你的名字：", "世界")

# 按钮
if st.button("打个招呼"):
    st.success(f"你好，{name}！欢迎使用 Streamlit 👋")

# 分割线
st.divider()

# 生成示例数据
st.subheader("📊 示例数据图表")
data = pd.DataFrame(
    np.random.randn(20, 2),
    columns=["A", "B"]
)

# 显示表格
st.dataframe(data)

# 显示折线图
st.line_chart(data)

# 侧边栏
st.sidebar.title("⚙️ 设置")
option = st.sidebar.selectbox(
    "选择一个选项：",
    ["选项一", "选项二", "选项三"]
)
st.sidebar.write("你选择了：", option)
