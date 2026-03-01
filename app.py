"""
Streamlit Web App — Candlestick Chart OHLC Extractor
Uses a trained YOLOv8 model to detect candlesticks and extract OHLC data.
"""

import sys
import streamlit as st
import cv2
import numpy as np
import pandas as pd
import tempfile
import os
from pathlib import Path
from PIL import Image
from streamlit_cropper import st_cropper

# ── Ensure project root is on sys.path (for src.chart_analyzer imports) ──────
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Candlestick OHLC Extractor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS loader ───────────────────────────────────────────────────────────────
STYLES_DIR = Path(__file__).parent / "styles"
CSS_FILES = [
    "global.css",
    "hero.css",
    "cards.css",
    "sidebar.css",
    "buttons.css",
    "components.css",
]


def load_css() -> None:
    for css_file in CSS_FILES:
        css_path = STYLES_DIR / css_file
        if css_path.exists():
            st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


load_css()

# ── Constants ────────────────────────────────────────────────────────────────
MODEL_PATH = str(Path(__file__).parent / "best.pt")


# ── Helpers ──────────────────────────────────────────────────────────────────

@st.cache_resource
def load_yolo_model(model_path: str):
    """Load the YOLO model once and cache it."""
    from ultralytics import YOLO
    return YOLO(model_path)


def detect_candlesticks(model, image_bgr: np.ndarray, conf_threshold: float = 0.25):
    """Run YOLO inference and return detections sorted left→right."""
    results = model(image_bgr, conf=conf_threshold, verbose=False)

    detections = []
    if len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes
        for i in range(len(boxes)):
            box = boxes.xyxy[i].cpu().numpy()
            confidence = float(boxes.conf[i].cpu().numpy())
            class_id = int(boxes.cls[i].cpu().numpy())
            x1, y1, x2, y2 = box
            detections.append({
                "x1": int(x1), "y1": int(y1),
                "x2": int(x2), "y2": int(y2),
                "width": int(x2 - x1),
                "height": int(y2 - y1),
                "center_x": int((x1 + x2) / 2),
                "center_y": int((y1 + y2) / 2),
                "confidence": confidence,
                "class_id": class_id,
            })
    detections.sort(key=lambda d: d["center_x"])
    return detections


def classify_candle_color(roi_bgr: np.ndarray) -> str:
    """Return 'bullish' or 'bearish' based on green/red pixel count."""
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

    green_mask = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
    red_mask1 = cv2.inRange(hsv, np.array([0, 40, 40]), np.array([10, 255, 255]))
    red_mask2 = cv2.inRange(hsv, np.array([160, 40, 40]), np.array([180, 255, 255]))
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    return "bullish" if np.sum(green_mask > 0) >= np.sum(red_mask > 0) else "bearish"


def extract_body_bounds(roi_bgr: np.ndarray):
    """Estimate body top/bottom within the ROI using horizontal projection."""
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    h_proj = np.sum(gray, axis=1).astype(float)
    if h_proj.max() == 0:
        return 0, roi_bgr.shape[0]
    threshold = h_proj.max() * 0.3
    above = np.where(h_proj > threshold)[0]
    if len(above) == 0:
        return 0, roi_bgr.shape[0]
    return int(above[0]), int(above[-1])


def extract_ohlc(detections, image_bgr, price_min, price_max):
    """Convert pixel detections → OHLC values using a linear price mapping."""
    if not detections:
        return []

    all_y1 = [d["y1"] for d in detections]
    all_y2 = [d["y2"] for d in detections]
    pixel_top = min(all_y1)
    pixel_bottom = max(all_y2)
    pixel_range = pixel_bottom - pixel_top
    if pixel_range == 0:
        pixel_range = 1

    def y_to_price(y_px):
        ratio = (y_px - pixel_top) / pixel_range
        return price_max - ratio * (price_max - price_min)

    ohlc_data = []
    for idx, det in enumerate(detections, start=1):
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        roi = image_bgr[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        direction = classify_candle_color(roi)
        body_top_rel, body_bot_rel = extract_body_bounds(roi)

        high = y_to_price(y1)
        low = y_to_price(y2)
        body_top_price = y_to_price(y1 + body_top_rel)
        body_bot_price = y_to_price(y1 + body_bot_rel)

        if direction == "bullish":
            open_price = body_bot_price
            close_price = body_top_price
        else:
            open_price = body_top_price
            close_price = body_bot_price

        ohlc_data.append({
            "Candle": idx,
            "Direction": direction,
            "Open": round(open_price, 2),
            "High": round(high, 2),
            "Low": round(low, 2),
            "Close": round(close_price, 2),
            "Confidence": round(det["confidence"], 4),
        })
    return ohlc_data


def draw_detections(image_bgr, detections):
    """Draw bounding boxes on the image and return an annotated copy."""
    vis = image_bgr.copy()
    for det in detections:
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        roi = image_bgr[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        direction = classify_candle_color(roi)
        color = (0, 200, 0) if direction == "bullish" else (0, 0, 200)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = f"{det['confidence']:.2f}"
        cv2.putText(vis, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
    return vis


# ── Main App ─────────────────────────────────────────────────────────────────
def main():
    # ── Hero header ──────────────────────────────────────────────────────────
    st.markdown("""
    <div class="hero-header">
        <h1>📈 <span class="hero-accent">Candlestick</span> OHLC Extractor</h1>
        <p>
            Upload a candlestick chart screenshot — from TradingView, Zerodha, or any broker —
            and extract <strong>Open · High · Low · Close</strong> data using a trained
            <strong>YOLOv8</strong> detection model.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-logo">
            <div class="logo-text">📊 OHLC Extractor</div>
            <div class="logo-sub">Powered by YOLOv8</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="sidebar-section-label">Detection</div>', unsafe_allow_html=True)
        conf_threshold = st.slider(
            "Confidence threshold",
            min_value=0.01, max_value=1.0, value=0.25, step=0.01,
            help="Lower = more detections (may include noise). Higher = fewer but more confident detections.",
        )

        st.markdown('<div class="sidebar-section-label">Price Axis</div>', unsafe_allow_html=True)
        use_price_mapping = st.toggle("Enable price-axis mapping", value=True)
        if use_price_mapping:
            col_a, col_b = st.columns(2)
            price_min = col_a.number_input("Min", value=0.0, format="%.2f", label_visibility="visible")
            price_max = col_b.number_input("Max", value=1000.0, format="%.2f", label_visibility="visible")
        else:
            price_min, price_max = 0.0, 1.0

        st.markdown('<div class="sidebar-section-label">Options</div>', unsafe_allow_html=True)
        enable_crop = st.toggle(
            "Crop image first",
            value=True,
            help="Draw a crop region on the uploaded image to focus only on the candle area.",
        )
        use_pipeline = st.toggle(
            "Full analysis pipeline",
            value=False,
            help="Runs the full ChartAnalyzer pipeline (TradingView preprocessing + OCR axis calibration). "
                 "Slower but attempts automatic price mapping.",
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="sidebar-section-label">Model Status</div>', unsafe_allow_html=True)

    # ── Model loading ─────────────────────────────────────────────────────────
    if not Path(MODEL_PATH).exists():
        st.error(f"⚠️  Model not found at `{MODEL_PATH}`. Please ensure **best.pt** is in the project root.")
        return

    model = load_yolo_model(MODEL_PATH)
    st.sidebar.success("✅  Model loaded: `best.pt`")

    # ── Image upload ──────────────────────────────────────────────────────────
    st.markdown("### 📂 Upload Chart Image")
    uploaded_file = st.file_uploader(
        "Drag & drop or click to browse",
        type=["png", "jpg", "jpeg", "bmp", "webp"],
        label_visibility="collapsed",
    )

    if uploaded_file is None:
        st.markdown("""
        <div class="upload-prompt">
            <div class="upload-icon">📤</div>
            <strong>Drop a candlestick chart image here</strong><br>
            <span style="font-size:0.85rem;">Supports PNG, JPG, BMP, WebP</span>
        </div>
        """, unsafe_allow_html=True)
        return

    # Read image
    file_bytes = np.frombuffer(uploaded_file.read(), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image_bgr is None:
        st.error("Could not decode the uploaded image.")
        return

    st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

    # ── Crop step ─────────────────────────────────────────────────────────────
    if enable_crop:
        st.markdown("### ✂️ Crop the Chart Area")
        st.caption(
            "Resize the selection box to include only the candlestick region — "
            "exclude toolbars, volume panels, and axis labels for best accuracy."
        )
        pil_image = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        cropped_pil = st_cropper(
            pil_image,
            realtime_update=True,
            box_color="#3a7bd5",
            aspect_ratio=None,
            return_type="image",
        )
        with st.expander("👁️  Preview cropped region", expanded=True):
            st.image(cropped_pil, caption="This region will be analysed", use_container_width=True)
        image_bgr = cv2.cvtColor(np.array(cropped_pil), cv2.COLOR_RGB2BGR)
    else:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        with st.expander("🖼️  Uploaded chart", expanded=True):
            st.image(image_rgb, caption=f"Uploaded: {uploaded_file.name}", use_container_width=True)

    st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

    # ── Run analysis ──────────────────────────────────────────────────────────
    if use_pipeline:
        with st.spinner("⚙️  Running full analysis pipeline (detection + OCR calibration)…"):
            results, annotated_img = run_full_pipeline(image_bgr, uploaded_file.name, conf_threshold)
        if results is None:
            return
        ohlc_data = results.get("market_data", [])
        display_data = []
        for dp in ohlc_data:
            display_data.append({
                "Candle": dp.get("id", 0),
                "Direction": dp.get("direction", "unknown"),
                "Open": round(dp.get("open", 0), 2),
                "High": round(dp.get("high", 0), 2),
                "Low": round(dp.get("low", 0), 2),
                "Close": round(dp.get("close", 0), 2),
                "Confidence": round(dp.get("confidence", 0), 4),
            })
        ohlc_df = pd.DataFrame(display_data) if display_data else pd.DataFrame()

        if annotated_img is not None:
            st.markdown("### 🔍 Detected Candlesticks")
            st.image(
                cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB),
                caption=f"Detections — {results.get('clean_detections', 0)} candlesticks found",
                use_container_width=True,
            )
    else:
        with st.spinner("🔎  Detecting candlesticks with YOLO…"):
            detections = detect_candlesticks(model, image_bgr, conf_threshold)

        if not detections:
            st.warning("⚠️  No candlesticks detected. Try lowering the confidence threshold.")
            return

        annotated = draw_detections(image_bgr, detections)
        st.markdown("### 🔍 Detected Candlesticks")
        st.image(
            cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
            caption=f"{len(detections)} candlestick{'s' if len(detections) != 1 else ''} detected",
            use_container_width=True,
        )

        if use_price_mapping and price_max > price_min:
            ohlc_list = extract_ohlc(detections, image_bgr, price_min, price_max)
        else:
            ohlc_list = extract_ohlc(detections, image_bgr, 0.0, 1.0)

        ohlc_df = pd.DataFrame(ohlc_list)

    # ── Display results ───────────────────────────────────────────────────────
    if ohlc_df.empty:
        st.warning("⚠️  No OHLC data could be extracted.")
        return

    st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)
    st.markdown("### 📊 Extracted OHLC Data")

    # Metric row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Candles", len(ohlc_df))
    if "Direction" in ohlc_df.columns:
        bull_count = int((ohlc_df["Direction"] == "bullish").sum())
        bear_count = int((ohlc_df["Direction"] == "bearish").sum())
        bull_pct = f"+{bull_count / len(ohlc_df):.0%}" if len(ohlc_df) else ""
        bear_pct = f"-{bear_count / len(ohlc_df):.0%}" if len(ohlc_df) else ""
        col2.metric("🟢 Bullish", bull_count, bull_pct)
        col3.metric("🔴 Bearish", bear_count, bear_pct)
    if "Confidence" in ohlc_df.columns:
        col4.metric("Avg Confidence", f"{ohlc_df['Confidence'].mean():.1%}")

    st.markdown("<br>", unsafe_allow_html=True)

    # Styled dataframe
    def _style_direction(val):
        if val == "bullish":
            return "color: #00c850; font-weight: 600;"
        elif val == "bearish":
            return "color: #ff4444; font-weight: 600;"
        return ""

    styled_df = (
        ohlc_df.style
        .applymap(_style_direction, subset=["Direction"] if "Direction" in ohlc_df.columns else [])
        .format({
            "Open": "{:.2f}", "High": "{:.2f}", "Low": "{:.2f}", "Close": "{:.2f}",
            "Confidence": "{:.2%}",
        }, na_rep="—")
        .set_properties(**{"text-align": "center"})
        .set_table_styles([
            {"selector": "thead th", "props": [
                ("background-color", "#1c2333"),
                ("color", "rgba(255,255,255,0.75)"),
                ("font-size", "0.8rem"),
                ("font-weight", "600"),
                ("text-transform", "uppercase"),
                ("letter-spacing", "0.06em"),
                ("border-bottom", "1px solid rgba(255,255,255,0.1)"),
                ("padding", "10px 12px"),
            ]},
            {"selector": "td", "props": [
                ("padding", "8px 12px"),
                ("border-bottom", "1px solid rgba(255,255,255,0.045)"),
                ("font-size", "0.9rem"),
            ]},
        ])
        .hide(axis="index")
    )
    st.dataframe(styled_df, use_container_width=True, hide_index=True)

    # ── Price chart ───────────────────────────────────────────────────────────
    if {"Open", "High", "Low", "Close"}.issubset(ohlc_df.columns) and len(ohlc_df) > 0:
        st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)
        st.markdown("### 📉 Reconstructed Price Chart")
        import plotly.graph_objects as go  # type: ignore

        fig = go.Figure(data=[go.Candlestick(
            x=ohlc_df["Candle"],
            open=ohlc_df["Open"],
            high=ohlc_df["High"],
            low=ohlc_df["Low"],
            close=ohlc_df["Close"],
            increasing_line_color="#00c850",
            increasing_fillcolor="rgba(0,200,80,0.75)",
            decreasing_line_color="#ff4444",
            decreasing_fillcolor="rgba(255,68,68,0.75)",
        )])
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(14,17,23,1)",
            font=dict(family="Inter, sans-serif", color="rgba(255,255,255,0.75)"),
            xaxis=dict(
                title="Candle #",
                gridcolor="rgba(255,255,255,0.05)",
                linecolor="rgba(255,255,255,0.1)",
                rangeslider=dict(visible=False),
            ),
            yaxis=dict(
                title="Price",
                gridcolor="rgba(255,255,255,0.05)",
                linecolor="rgba(255,255,255,0.1)",
                side="right",
            ),
            height=480,
            margin=dict(l=20, r=60, t=30, b=50),
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Download ──────────────────────────────────────────────────────────────
    st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)
    csv_data = ohlc_df.to_csv(index=False)
    dl_col, _ = st.columns([1, 3])
    dl_col.download_button(
        label="⬇️  Download OHLC as CSV",
        data=csv_data,
        file_name="ohlc_data.csv",
        mime="text/csv",
        use_container_width=True,
    )


def run_full_pipeline(image_bgr, filename, conf_threshold):
    """Run the full ChartAnalyzer pipeline by writing to a temp file."""
    try:
        from src.chart_analyzer.analyzer import ChartAnalyzer

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            cv2.imwrite(tmp.name, image_bgr)
            tmp_path = tmp.name

        analyzer = ChartAnalyzer(
            tmp_path,
            use_yolo=True,
            yolo_model_path=MODEL_PATH,
        )
        if hasattr(analyzer.detector, "yolo_detector"):
            analyzer.detector.yolo_detector.confidence_threshold = conf_threshold

        results = analyzer.analyze()

        annotated_img = None
        raw_dets = analyzer.detector.detect_raw(Path(tmp_path))
        clean_dets = analyzer.detector.clean_detections(raw_dets)
        if clean_dets:
            annotated_img = image_bgr.copy()
            for det in clean_dets:
                x, y, w, h = det["x"], det["y"], det["width"], det["height"]
                cv2.rectangle(annotated_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        os.unlink(tmp_path)
        return results, annotated_img

    except Exception as e:
        st.error(f"Pipeline error: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None, None


if __name__ == "__main__":
    main()
