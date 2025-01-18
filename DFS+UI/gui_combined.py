import streamlit as st
import os
from PIL import Image
from quality_score import evaluate_image
from metric import process_image_and_calculate_metrics
import numpy as np

# Đặt cấu hình trang phải ở dòng đầu tiên
st.set_page_config(page_title="Iris Quality Assessment", layout="wide")

# Giao diện Streamlit
st.title("Iris Quality Assessment")
st.markdown(
    """
    <style>
        .main {
            background-color: #f9f9f9;
            border: 1px solid #ddd;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
        }
        h1 {
            color: #4CAF50;
            text-align: center;
        }
        .score {
            font-size: 24px;
            font-weight: bold;
            color: #4CAF50;
            text-align: center;
        }
        .evaluation {
            text-align: center;
            margin-top: 10px;
            font-size: 18px;
            font-weight: bold;
        }
        .evaluate-button {
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            font-size: 16px;
            cursor: pointer;
        }
        .evaluate-button:hover {
            background-color: #45a049;
        }
        .metric-container {
            margin: 0 auto;
            padding: 20px;
            background-color: #fff;
            border: 1px solid #ddd;
            border-radius: 10px;
            box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
            max-width: 600px;
            text-align: center;
        }
        .metric-item {
            display: inline-flex;
            justify-content: center;
            padding: 5px 0;
            border-bottom: 1px solid #ddd;
            width: 100%;
        }
        .metric-item:last-child {
            border-bottom: none;
        }
        .metric-key {
            font-weight: bold;
            color: #FFD700;
            margin-right: 20px;
            text-align: left;
        }
        .metric-value {
            color: #FFD700;
            text-align: left;
        }
    </style>
    """,
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    temp_dir = "temp_images"
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, uploaded_file.name)
    image.save(temp_path)

    if st.button("DFS Mode", key="dfs_mode", help="Evaluate using DFS mode"):
        score = evaluate_image(temp_path)

        st.markdown("### DFS Evaluation Result")
        st.image(image, caption="Uploaded Image", use_container_width=True)
        st.markdown(f"<div class='score'>Quality Score: {score}</div>", unsafe_allow_html=True)
        st.progress(score / 100)

        if score > 98:
            st.markdown("<div class='evaluation' style='color: green;'>Good Quality</div>", unsafe_allow_html=True)
        elif score > 90:
            st.markdown("<div class='evaluation' style='color: orange;'>Average Quality</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='evaluation' style='color: red;'>Poor Quality</div>", unsafe_allow_html=True)

    if st.button("Hand-crafted Mode", key="hand_crafted_mode", help="Evaluate using Hand-crafted metrics"):
        # Tính toán các chỉ số và lấy ảnh mask
        metrics, mask_pupil_resized = process_image_and_calculate_metrics(temp_path)

        # Chuyển ma trận mask về ảnh PIL
        mask_pupil_resized = (mask_pupil_resized * 255 / np.max(mask_pupil_resized)).astype(np.uint8)
        mask_pupil_resized = Image.fromarray(mask_pupil_resized)

        # Đặt ngưỡng cố định
        thresholds = {
            "Pixel Count Score": (5, lambda x: x >= 5),
            "Sharpness Score": (12, lambda x: x >= 12),
            "Off-Angle Score": (20, lambda x: x <= 20),
            "Dilation Score": (75, lambda x: x >= 75),
            "GLS Score": (5, lambda x: x >= 5),
        }

        # Hiển thị tiêu đề
        st.markdown("### Hand-crafted Metrics Evaluation")

        # Hiển thị ảnh gốc và ảnh mask cạnh nhau
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)
        with col2:
            st.image(mask_pupil_resized, caption="Mask Image", use_container_width=True, clamp=True)

        # Hiển thị các chỉ số đánh giá
        st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
        violation_found = False
        for metric, value in metrics.items():
            threshold_value, condition = thresholds.get(metric, (None, lambda x: True))
            is_valid = condition(value)  # Kiểm tra ngưỡng
            color = "green" if is_valid else "red"
            if not is_valid:
                violation_found = True

            # Hiển thị giá trị thực và ngưỡng
            st.markdown(
                f"<div class='metric-item'><span class='metric-key'>{metric}</span><span class='metric-value' style='color: {color};'>{value:.2f} (Threshold: {threshold_value})</span></div>",
                unsafe_allow_html=True
            )

        st.markdown("</div>", unsafe_allow_html=True)

        # Hiển thị đánh giá tổng thể
        if violation_found:
            st.markdown("<div class='evaluation' style='color: red;'>Poor Image</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='evaluation' style='color: green;'>Good Image</div>", unsafe_allow_html=True)



    # Xóa ảnh tạm thời khi kết thúc
    if st.button("Clear Temporary Files"):
        os.remove(temp_path)
        st.info("Temporary file cleared!")
