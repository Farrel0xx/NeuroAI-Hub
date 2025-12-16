import os
import numpy as np
import tensorflow as tf
import streamlit as st
import scipy.ndimage
import cv2
from PIL import Image
import io  # <-- TAMBAHAN: Untuk mengubah gambar ke bytes
import base64  # Tambahan untuk display GIF

# --- GEMINI IMPORT ---
from google import genai
from google.genai import types

# --- Initial Configuration ---
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Use CPU only
MODEL_PATH = "brain_tumor_segmentation.h5"
IMAGE_WIDTH_PX = 450  # Slightly larger for better visuals

# ========================================
# === ALL MODEL FUNCTIONS ===
# ========================================

def tversky_loss(y_true, y_pred, alpha=0.7, beta=0.3):
    smooth = tf.keras.backend.epsilon()
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.clip_by_value(y_pred, smooth, 1.0 - smooth)
    tp = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    fp = tf.reduce_sum((1 - y_true) * y_pred, axis=[1, 2, 3])
    fn = tf.reduce_sum(y_true * (1 - y_pred), axis=[1, 2, 3])
    tversky_index = (tp + smooth) / (tp + alpha * fn + beta * fp + smooth)
    return tf.reduce_mean(1 - tversky_index)

def focal_tversky_loss(y_true, y_pred, alpha=0.7, beta=0.3, gamma=1.3):
    tversky = tversky_loss(y_true, y_pred, alpha, beta)
    focal_tversky = tf.math.log(tf.math.cosh(tversky * gamma))
    return focal_tversky

def combined_loss(y_true, y_pred, alpha=0.7, beta=0.3, gamma=1.3, weight=0.5):
    dice = 1 - (2 * tf.reduce_sum(y_true * y_pred) + 1e-6) / \
           (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + 1e-6)
    tversky = focal_tversky_loss(y_true, y_pred, alpha, beta, gamma)
    return weight * dice + (1 - weight) * tversky

@st.cache_resource
def load_segmentation_model(path):
    try:
        model = tf.keras.models.load_model(
            path,
            custom_objects={
                "tversky_loss": tversky_loss,
                "focal_tversky_loss": focal_tversky_loss,
                "combined_loss": combined_loss,
            },
            compile=False
        )
        model.compile(optimizer="adam", loss=combined_loss)
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        return None

def postprocess_mask(mask, threshold=0.10, kernel_size=3):
    mask = mask > threshold
    mask = mask.squeeze()
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    mask = scipy.ndimage.binary_dilation(mask, structure=kernel)
    mask = scipy.ndimage.binary_erosion(mask, structure=kernel)
    return mask.astype(np.float32)

def generate_heatmap(mask, image_rgb):
    mask_resized = cv2.resize(mask, (image_rgb.shape[1], image_rgb.shape[0]))
    mask_uint8 = (mask_resized * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(mask_uint8, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(image_rgb, 0.65, heatmap, 0.35, 0)
    
    tumor_pixels = np.count_nonzero(mask_resized)
    total_pixels = mask_resized.size
    tumor_ratio = tumor_pixels / total_pixels

    if tumor_ratio < 0.1:
        severity = "⚠️ Mild"
    elif 0.1 <= tumor_ratio < 0.3:
        severity = "‼️ Moderate"
    else:
        severity = "‼️ Severe"

    return overlay, severity, tumor_ratio * 100

def segment_image(model, image_array):
    # Ensure image_array is RGB
    if image_array.ndim == 2:
        image_array = cv2.cvtColor(image_array.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    image_tensor = tf.convert_to_tensor(image_array, dtype=tf.float32)
    image_resized = tf.image.resize(image_tensor, (128, 128))
    image_resized = np.expand_dims(image_resized, axis=0) / 255.0

    if model is None:
        raise ValueError("Model failed to load.")

    prediction = model.predict(image_resized, verbose=0)
    mask = postprocess_mask(prediction[0], threshold=0.15)

    if mask.ndim == 2:
        mask = np.expand_dims(mask, axis=-1)

    height, width = image_array.shape[:2]
    mask_resized_output = tf.image.resize(mask, (height, width)).numpy().squeeze()

    heatmap_overlay, severity, tumor_ratio_percent = generate_heatmap(mask_resized_output, image_array)

    return mask_resized_output, heatmap_overlay, severity, tumor_ratio_percent

# === MODIFIED FUNCTION FOR MULTIMODAL INPUT ===
def get_ai_interpretation(api_key, original_image_array, heatmap_array, mask_array_255, severity, tumor_ratio):
    if not api_key:
        return "⚠️ Gemini API Key not provided. AI interpretation disabled."

    try:
        client = genai.Client(api_key=api_key)
        GEMINI_MODEL = 'gemini-2.5-flash'
    except Exception as e:
        return f"❌ Gemini initialization failed: {e}"

    # --- 1. Helper Function for Conversion ---
    def array_to_image_part(img_array, mime_type="image/jpeg"):
        # Ensure array is in uint8 format for PIL
        img_pil = Image.fromarray(img_array.astype(np.uint8))
        img_bytes = io.BytesIO()
        img_pil.save(img_bytes, format='JPEG')
        return types.Part.from_bytes(data=img_bytes.getvalue(), mime_type=mime_type)

    # --- 2. Convert and Prepare Parts ---
    
    # Mask array might be grayscale, convert to 3-channel for better visual interpretation by Gemini
    if mask_array_255.ndim == 2:
        mask_array_rgb = cv2.cvtColor(mask_array_255.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    else:
        mask_array_rgb = mask_array_255

    original_part = array_to_image_part(original_image_array)
    heatmap_part = array_to_image_part(heatmap_array)
    mask_part = array_to_image_part(mask_array_rgb)

    # --- 3. Prompt that requests analysis of all three images ---
    prompt = f"""
    You are a professional AI radiologist with extensive experience.
    This analysis is based on Deep Learning segmentation results from a brain MRI scan.
    
    You have been provided with THREE images for analysis:
    1. The original MRI scan.
    2. The segmentation Heatmap Overlay (Red/Orange = high tumor probability).
    3. The Binary Segmentation Mask (White = detected tumor region).

    Key findings from the DL model:
    - Severity Level : {severity}
    - Tumor Area Ratio: {tumor_ratio:.2f}%

    Provide a concise, professional, and easy-to-understand interpretation in English.
    Crucially, use the Heatmap and Mask to visually confirm the segmentation and comment on the tumor's appearance, shape, and localization.

    • Summary of main findings, confirming the tumor area seen in the mask and heatmap.
    • Analysis of the Tumor Ratio and its clinical implication.
    • Recommended next steps based on severity level.

    Use clean bullet-point formatting.
    """

    try:
        # Send prompt and all three image parts
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[prompt, original_part, heatmap_part, mask_part]
        )
        return response.text
    except Exception as e:
        return f"❌ Gemini API Error: {e}"

# ========================================
# === FUNCTION TO GENERATE GIF FROM DICOM SLICES WITH SEGMENTATION ===
# ========================================
import pydicom
import imageio

def generate_gif_from_dicom(model, dicom_files, output_gif='animation.gif', gemini_api_key=None):
    slices = []
    for file in dicom_files:
        try:
            ds = pydicom.dcmread(io.BytesIO(file.getvalue()))
            key = ds.InstanceNumber if hasattr(ds, 'InstanceNumber') else (ds.SliceLocation if hasattr(ds, 'SliceLocation') else 0)
            slices.append((key, ds))
        except:
            pass

    if len(slices) == 0:
        st.error("No valid DICOM files found.")
        return None, None, None

    slices.sort(key=lambda x: x[0])
    datasets = [s[1] for s in slices]

    images = []
    tumor_ratios = []
    severities = []
    representative_original = None
    representative_heatmap = None
    representative_mask = None
    middle_index = len(datasets) // 2

    for i, ds in enumerate(datasets):
        img = ds.pixel_array
        low, high = np.percentile(img, (1, 99))
        img = np.clip(img, low, high)
        img = (img - img.min()) / (img.max() - img.min() + 1e-6) * 255
        img = img.astype(np.uint8)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        mask, heatmap, severity, tumor_ratio = segment_image(model, img_rgb)
        images.append(heatmap)  # Use segmented heatmap as frame
        tumor_ratios.append(tumor_ratio)
        severities.append(severity)

        if i == middle_index:
            representative_original = img_rgb
            representative_heatmap = heatmap
            representative_mask = (mask * 255).astype(np.uint8)

    images += images[-2::-1]  # Bolak-balik for smooth animation

    # Save GIF with infinite loop
    imageio.mimsave(output_gif, images, duration=0.1, loop=0)

    # Calculate average metrics
    avg_tumor_ratio = np.mean(tumor_ratios)
    overall_severity = max(set(severities), key=severities.count)  # Most common severity

    # Get AI report for representative slice if API key provided
    ai_report = None
    if gemini_api_key:
        ai_report = get_ai_interpretation(
            gemini_api_key,
            representative_original,
            representative_heatmap,
            representative_mask,
            overall_severity,
            avg_tumor_ratio
        )

    return output_gif, overall_severity, avg_tumor_ratio, ai_report

# ========================================
# === FUNCTION TO DISPLAY ANIMATED GIF ===
# ========================================
def display_gif(gif_path):
    with open(gif_path, 'rb') as f:
        gif_bytes = f.read()
    base64_gif = base64.b64encode(gif_bytes).decode('utf-8')
    st.markdown(f'<img src="data:image/gif;base64,{base64_gif}" alt="brain tumor animation" width="{IMAGE_WIDTH_PX}">', unsafe_allow_html=True)

# ========================================
# === ENHANCED CSS: EVEN COOLER, MODERN FUTURISTIC THEME ===
# ========================================

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Orbitron:wght@500;700&display=swap');

html, body, [class*="stApp"] {
    font-family: 'Inter', sans-serif;
    background: linear-gradient(135deg, #0a0e17 0%, #1a2332 50%, #2d3748 100%);
    color: #e2e8f0;
}

h1 {
    font-family: 'Orbitron', sans-serif;
    color: transparent;
    background: linear-gradient(to right, #22d3ee, #4ade80);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 3.5rem;
    font-weight: 700;
    text-align: center;
    text-shadow: 0 0 20px rgba(34, 211, 238, 0.5);
    margin-bottom: 1.5rem;
}

h2, h3, h4, h5 {
    color: #a5f3fc;
    font-weight: 600;
}

[data-testid="stSidebar"] {
    background: linear-gradient(to bottom, #111827, #0a0e17);
    border-right: 1px solid #334155;
    box-shadow: 8px 0 30px rgba(0,0,0,0.8);
}

/* LOGO: CENTERED, NEON GLOW, PREMIUM */
[data-testid="stSidebar"] .stImage > div {
    display: flex !important;
    justify-content: center !important;
    margin: 40px 0 !important;
}

[data-testid="stSidebar"] .stImage img {
    border-radius: 50% !important;
    border: 4px solid #22d3ee !important;
    box-shadow: 0 0 50px rgba(34, 211, 238, 0.8) !important;
    max-width: 180px !important;
}

/* CENTER ALL SIDEBAR TEXT */
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] * {
    text-align: center !important;
}

[data-testid="stMetric"] {
    background: rgba(15, 23, 42, 0.7);
    border-radius: 18px;
    padding: 25px;
    border: 1px solid #22d3ee;
    box-shadow: 0 10px 30px rgba(34, 211, 238, 0.2);
    backdrop-filter: blur(10px);
}

.ratio-box {
    background: linear-gradient(to right, #1e3a8a, #22d3ee);
    padding: 20px;
    border-radius: 18px;
    color: white;
    font-weight: 700;
    font-size: 1.3rem;
    text-align: center;
    box-shadow: 0 8px 25px rgba(34, 211, 238, 0.4);
}

.doctor-box {
    background: rgba(10, 14, 23, 0.95);
    border: 2px solid #22d3ee;
    border-radius: 20px;
    padding: 35px;
    box-shadow: 0 15px 40px rgba(34, 211, 238, 0.3);
    backdrop-filter: blur(12px);
    margin: 30px 0;
}

.doctor-box h5 {
    color: #22d3ee;
    border-bottom: 2px solid #22d3ee;
    padding-bottom: 12px;
}

button[kind="primary"] {
    background: linear-gradient(to right, #22d3ee, #4ade80) !important;
    border: none !important;
    border-radius: 16px !important;
    padding: 16px 32px !important;
    font-weight: 700 !important;
    box-shadow: 0 10px 30px rgba(34, 211, 238, 0.5) !important;
}

button[kind="primary"]:hover {
    transform: translateY(-5px);
    box-shadow: 0 15px 40px rgba(34, 211, 238, 0.7) !important;
}

figcaption {
    color: #94a3b8;
    font-style: italic;
    text-align: center;
    margin-top: 12px;
}

@media (max-width: 768px) {
    h1 { font-size: 2.6rem; }
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ========================================
# === STREAMLIT UI ===
# ========================================

st.set_page_config(
    page_title="NeuroAI Brain Tumor Detector",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Sidebar with more explanations ---
with st.sidebar:
    # Premium glowing neon brain logo (transparent PNG)
    st.image("https://thumbs.dreamstime.com/b/futuristic-glowing-ai-brain-made-neon-circuits-digital-data-streams-set-dark-gradient-background-visually-356224505.jpg", width=200)
    
    st.title("🧠 NeuroAI Hub")
    
    st.markdown("""
    **Welcome to NeuroAI Hub** - Advanced AI-Powered Medical Imaging Solutions

    This platform provides cutting-edge brain MRI analysis using Deep Learning (U-Net architecture) to detect and segment brain tumors with high precision. It includes stunning visualizations, AI-powered radiologist interpretations, and animated DICOM series analysis.
    """)
    
    st.markdown("### Key Features")
    st.markdown("""
    - **Tumor Segmentation**: Uses U-Net model for accurate tumor detection.
    - **Heatmap Overlays**: Visualizes tumor probability with color-coded maps.
    - **Severity Assessment**: Classifies tumor severity based on area ratio.
    - **AI Interpretation**: Powered by Gemini AI for professional radiological reports.
    - **DICOM Animation**: Creates looping GIFs with segmented overlays from DICOM series.
    - **Multi-Slice Analysis**: Processes entire series for comprehensive insights.
    """)
    
    st.markdown("### How It Works")
    st.markdown("""
    1. Upload a single MRI image or multiple DICOM files.
    2. The U-Net model segments potential tumor regions.
    3. Generate heatmaps, masks, and metrics.
    4. For DICOM series, create animated GIF with segmentation.
    5. Gemini AI provides detailed analysis and recommendations.
    """)
    
    st.markdown("### Important Notes")
    st.markdown("""
    - This tool is for educational and research purposes only. Not a substitute for professional medical advice.
    - Ensure images are anonymized and comply with privacy regulations.
    - Model accuracy depends on training data; always consult a healthcare professional.
    - For best results, use high-quality MRI scans in supported formats.
    """)
    
    st.markdown("### Advanced Usage")
    st.markdown("""
    - **Single Image**: Upload JPG/PNG for static analysis.
    - **DICOM Series**: Upload multiple .dcm files for animated visualization.
    - **API Integration**: Enter Gemini API key for enhanced AI reports.
    - **Customization**: Adjust thresholds in code for fine-tuning (advanced users).
    """)
    
    st.markdown("---")
    
    st.subheader("🔑 Gemini API Key")
    gemini_api_key = st.text_input("Enter API Key:", type="password", placeholder="AIzaSy...")
    st.markdown("""
    <small>⚠️ Get your API key from <a href="https://ai.google.dev" target="_blank">Google AI Studio</a></small>
    """, unsafe_allow_html=True)
    st.markdown("---")
    
    model = load_segmentation_model(MODEL_PATH)
    
    if model:
        st.success("✅ U-Net Model Loaded Successfully")
        if gemini_api_key:
            st.success("✨ Gemini AI Interpretation Enabled (Multimodal)")
        else:
            st.info("Enter API Key to enable AI report")
    else:
        st.error("❌ Model not found – ensure .h5 file exists")
    
    st.markdown("---")
    st.caption("***📌 Created by Farrel0xx • Built on Streamlit • ✨ Analysis by Gemini AI***")

# --- Main App ---
st.title("🧠 NeuroAI: Advanced Brain Tumor Detection")

st.markdown("""
**Upload a brain MRI image** for state-of-the-art tumor segmentation using Deep Learning, complete with premium visualizations and AI-powered radiologist interpretation.

This section allows analysis of single static images. For dynamic series, use the DICOM Animation section below.
""")

st.info("""
**Guide**: Upload high-resolution brain MRI scans in axial, coronal, or sagittal views. The system will automatically detect and highlight potential tumor regions.
""")

# Example MRI images for better UX
col_ex1, col_ex2 = st.columns(2)
with col_ex1:
    st.image("https://www.researchgate.net/publication/362650590/figure/fig2/AS:11431281085480726@1663779688380/The-Flair-T1-T1ce-and-T2-channels-of-the-brain-tumor-were-displayed-using-the.jpg", caption="**Example: Multi-modal MRI (FLAIR, T1, T1ce, T2) with tumor regions**", width=400)
with col_ex2:
    st.image("https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fs41598-025-21255-4/MediaObjects/41598_2025_21255_Fig1_HTML.png", caption="**Example: Brain tumor in multi-modal slices (T1, T1ce, T2, FLAIR + segmentation)**", width=400)

uploaded_file = st.file_uploader("Choose an MRI image (.jpg, .jpeg, .png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_array = np.array(image)
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("📷 Original MRI Image")
        st.image(image_array, width=IMAGE_WIDTH_PX)
    
    if model and st.button("▶️ Run Segmentation Analysis", type="primary", use_container_width=True):
        st.markdown("---")
        
        try:
            with st.spinner("🔬 Processing image and generating segmentation..."):
                mask_resized, heatmap_overlay, severity, tumor_ratio = segment_image(model, image_array)

            col_m, col_r = st.columns(2)
            with col_m:
                st.metric("***Severity Level***", severity)
            with col_r:
                st.markdown(f'<div class="ratio-box">Tumor Area Ratio<br>{tumor_ratio:.2f}%</div>', unsafe_allow_html=True)

            if "Severe" in severity:
                st.error("🚨 **CRITICAL:** Severe tumor detected – immediate medical consultation required!")
            elif "Moderate" in severity:
                st.warning("⚠️ **ATTENTION:** Moderate tumor detected – follow-up examination recommended")
            else:
                st.success("✅ **SAFE:** Mild findings – continue routine monitoring")

            st.markdown("---")

            if gemini_api_key:
                with st.spinner("👩‍⚕️ AI Radiologist analyzing results..."):
                    ai_report = get_ai_interpretation(
                        gemini_api_key, 
                        image_array, 
                        heatmap_overlay, 
                        (mask_resized * 255).astype(np.uint8),
                        severity, 
                        tumor_ratio
                    )
                
                st.markdown("### 👩‍⚕️ AI Radiologist Report (Gemini 2.5 Flash)")
                
                st.markdown(f"""
                <div class="doctor-box">
                {ai_report}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("Enable Gemini in sidebar for detailed AI interpretation report.")

            st.markdown("---")

            col_h, col_msk = st.columns(2)
            with col_h:
                st.subheader("🌡️ Tumor Heatmap Overlay")
                st.image(heatmap_overlay, caption="Red/orange areas = high tumor probability", width=IMAGE_WIDTH_PX)
            with col_msk:
                st.subheader("🎭 Binary Segmentation Mask")
                st.image((mask_resized * 255).astype(np.uint8), caption="White = detected tumor region", width=IMAGE_WIDTH_PX)

            with st.expander("📚 Technical Explanation"):
                st.markdown("""
                - **Binary Mask**: White pixels represent areas classified as tumor by the U-Net model.
                - **Heatmap Overlay**: Combines original image with JET colormap to highlight tumor boundaries.
                - **Severity & Ratio**: Automatically calculated based on tumor area percentage relative to total image.
                - **Model**: Trained U-Net with combined Tversky + Dice loss for optimal segmentation performance.
                - **Post-processing**: Includes dilation and erosion for smoother masks.
                """)

        except Exception as e:
            st.error(f"Analysis error: {e}")
else:
    st.info("👆 Upload a brain MRI image to start the analysis.")

# --- New Section for DICOM Animation ---
st.markdown("---")
st.title("📽️ Brain Tumor DICOM Animation")

st.markdown("""
Upload multiple DICOM files (.dcm) from a series (e.g., coronal or sagittal slices) to generate and display an animated GIF showing slice-by-slice tumor visualization with segmentation overlays.

This feature processes each slice individually, applies tumor segmentation, and creates a continuously looping animation for dynamic viewing.
""")

st.info("""
**Guide**: Upload a series of DICOM files from the same scan. The system will sort them, segment tumors, and generate an infinite-looping GIF with heatmaps. AI analysis is provided for the entire series based on a representative slice.
""")

uploaded_dicom_files = st.file_uploader("Choose DICOM files (.dcm)", type=["dcm"], accept_multiple_files=True)

if uploaded_dicom_files:
    if model and st.button("▶️ Generate and Display Animation", type="primary", use_container_width=True):
        with st.spinner("🔬 Generating segmented animation from DICOM slices..."):
            gif_path, overall_severity, avg_tumor_ratio, ai_report = generate_gif_from_dicom(model, uploaded_dicom_files, gemini_api_key=gemini_api_key)
            if gif_path:
                st.subheader("🎥 DICOM Slice Animation with Segmentation")
                display_gif(gif_path)
                
                col_m, col_r = st.columns(2)
                with col_m:
                    st.metric("***Overall Severity***", overall_severity)
                with col_r:
                    st.markdown(f'<div class="ratio-box">Average Tumor Ratio<br>{avg_tumor_ratio:.2f}%</div>', unsafe_allow_html=True)
                
                if ai_report:
                    st.markdown("### 👩‍⚕️ AI Analysis of DICOM Series (Gemini 2.5 Flash)")
                    st.markdown(f"""
                    <div class="doctor-box">
                    {ai_report}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info("Enable Gemini API for series analysis.")
                
                st.markdown("**Note:** The animation scrolls through slices forward and backward for smooth viewing. Includes segmentation overlays (red/orange for tumors). GIF loops continuously.")
                
                with st.expander("📚 DICOM Animation Explanation"):
                    st.markdown("""
                    - **Segmentation per Slice**: Each frame shows the heatmap overlay on the original slice.
                    - **Sorting**: Slices are sorted by Instance Number or Slice Location.
                    - **Normalization**: Images are contrast-adjusted for better visibility.
                    - **Looping**: Infinite loop for continuous replay.
                    - **Metrics**: Average tumor ratio and majority severity across all slices.
                    - **AI Report**: Based on a representative middle slice, providing insights for the series.
                    """)
