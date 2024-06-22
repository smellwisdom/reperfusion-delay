import streamlit as st
import shap
import pandas as pd
import joblib
import numpy as np
import streamlit.components.v1 as components
from scipy.special import expit  # 用于sigmoid函数

# 自定义CSS来扩大内容宽度
st.markdown(
    """
    <style>
    .main .block-container {
        max-width: 90%;
        padding-left: 5%;
        padding-right: 5%;
    }
    .force-plot {
        border: none;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# 加载训练好的模型
model_path = 'final_model.pkl'  # 更新模型路径为实际保存的模型文件名
loaded_model = joblib.load(model_path)

# 特征名称
feature_names = ['Door_to_Puncture_Time', 'Admission_mRS_Score', 'Antiplatelet_Therapy',
                 'Onset_to_door_Time', 'PreTreat_TICI_Grade', 'Mode_of_Arrival',
                 'BMI', 'TOAST_Classification', 'Age', 'PreTreat_NIHSS_Score', 'Systolic_BP']

# 映射关系
mode_of_arrival_map = {
    "In-hospital ambulance": 1,
    "Local emergency services": 2,
    "Inter-hospital transfer": 3,
    "Self-admission": 4
}

toast_classification_map = {
    "LAA": 1,
    "CE": 2,
    "SAA": 3,
    "SUE": 5
}

pre_treat_tici_grade_map = {
    "0": 1,
    "1": 2,
    "2a": 3,
    "2b": 4,
    "3": 5
}

# Streamlit 应用程序接口
# st.title("Patient reperfusion delay Prediction")

st.markdown("""
<style>
.big-font {
    font-size:20px !important;
    font-weight: bold;
}
.centered {
    display: flex;
    justify-content: center;
    align-items: center;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-font">Input Patient Details:</p>', unsafe_allow_html=True)

# 创建三个列来布局输入表单
col1, col2 = st.columns(2)

with col1:
    input_data = {}
    input_data['Door_to_Puncture_Time'] = st.slider('Door to Puncture Time (min)', min_value=4, max_value=1255, value=30, step=1)
    input_data['Admission_mRS_Score'] = st.selectbox('Admission mRS Score', options=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], index=2)
    input_data['Antiplatelet_Therapy'] = st.selectbox('Antiplatelet Therapy', options=['No', 'Yes'], index=1)
    input_data['PreTreat_TICI_Grade'] = st.selectbox('PreTreat TICI Grade', options=['0', '1', '2a', '2b', '3'], index=1)
    input_data['Mode_of_Arrival'] = st.selectbox('Mode of Arrival', options=list(mode_of_arrival_map.keys()), index=0)
    input_data['Onset_to_door_Time'] = st.slider('Onset to Door Time (min)', min_value=0, max_value=7200, value=45, step=1)
with col2:
    input_data['BMI'] = st.slider('BMI', min_value=15.24, max_value=41.52, value=25.0, step=0.1)
    input_data['TOAST_Classification'] = st.selectbox('TOAST Classification', options=list(toast_classification_map.keys()), index=1)
    input_data['Age'] = st.slider('Age', min_value=0, max_value=89, value=60, step=1)
    input_data['PreTreat_NIHSS_Score'] = st.slider('PreTreat NIHSS Score', min_value=0, max_value=40, value=8, step=1)
    input_data['Systolic_BP'] = st.slider('Systolic BP', min_value=88, max_value=240, value=140, step=1)

# 将输入数据转换为 DataFrame
input_df = pd.DataFrame([input_data])

# 处理映射关系
input_df['Antiplatelet_Therapy'] = input_df['Antiplatelet_Therapy'].map({'No': 0.0, 'Yes': 1.0})
input_df['Mode_of_Arrival'] = input_df['Mode_of_Arrival'].map(mode_of_arrival_map)
input_df['TOAST_Classification'] = input_df['TOAST_Classification'].map(toast_classification_map)
input_df['PreTreat_TICI_Grade'] = input_df['PreTreat_TICI_Grade'].map(pre_treat_tici_grade_map)

# 确保列顺序与模型训练时相同
input_df = input_df[feature_names]

# 进行预测
if st.button('Predict'):
    prediction_prob = loaded_model.predict_proba(input_df)[0, 1]
    # st.markdown('<p class="big-font">Prediction Result:</p>', unsafe_allow_html=True)
    # st.write(f"Based on feature values, predicted probability of HI is {prediction_prob * 100:.2f}%")
    # 使用 HTML 和内联 CSS 来设置文本的样式
    prediction_text = f"Based on feature values, predicted probability of surgical delay is {prediction_prob * 100:.2f}%"
    st.markdown(f'<p style="font-size:45px; font-weight: bold;">{prediction_text}</p>',
                unsafe_allow_html=True)

    # 创建 SHAP 解释器
    explainer = shap.Explainer(loaded_model, feature_names=feature_names)
    shap_values = explainer(input_df)

    # 如果基线值是标量，使用它，否则选择索引0或1
    base_value = explainer.expected_value
    if isinstance(base_value, np.ndarray) and len(base_value) > 1:
        base_value = base_value[0]

    # 创建 Force Plot 并保存为HTML文件
    shap.initjs()
    force_plot = shap.force_plot(base_value, shap_values.values[0], input_df, feature_names=feature_names, link='logit')

    # 转换SHAP值为概率值
    prob_value = expit(base_value + shap_values.values[0].sum())

    shap.save_html("force_plot.html", force_plot)

    # 读取HTML文件并在Streamlit中显示
    force_plot_html = ""
    with open("force_plot.html", "r", encoding="utf-8") as f:
        force_plot_html = f.read()

    # 使用 Streamlit 显示 HTML 力图，并使其居中显示
    st.markdown('<div class="centered">', unsafe_allow_html=True)
    components.html(f'<div class="force-plot">{force_plot_html}</div>', height=800, width=1700, scrolling=True)  # 调整高度和宽度确保显示完全
    st.markdown('</div>', unsafe_allow_html=True)
