# 📈 Candlestick Chart OHLC Extractor — Streamlit Web App

A web application that lets you upload a candlestick chart screenshot, crop it to the chart area, and extract **Open / High / Low / Close (OHLC)** data using a trained **YOLOv8** model.

---

## Folder Structure

```
streamlit_app/
├── app.py        # Main Streamlit application
├── best.pt       # Trained YOLOv8 model weights
└── README.md     # This file
```

The app also uses `src/chart_analyzer/` from the project root for the optional full pipeline mode.

---

## Requirements

Install all dependencies from the project root:

```bash
pip install -r requirements.txt
pip install streamlit streamlit-cropper plotly
```

---

## Running the App

### From the project root (recommended)

```bash
streamlit run app.py
```

### Directly from this folder

```bash
streamlit run streamlit_app/app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## How to Use

1. **Upload** a candlestick chart screenshot (PNG, JPG, WEBP, BMP).
2. **Crop** — drag the blue handles to select only the candle area (exclude toolbars, volume panel, axis labels). Toggle off in the sidebar to skip.
3. **Run detection** — the YOLOv8 model (`best.pt`) detects each candlestick automatically.
4. **View results**:
   - Annotated image with bounding boxes (green = bullish, red = bearish)
   - OHLC data table with candle direction and confidence score
   - Interactive Plotly candlestick chart
5. **Download** the extracted OHLC data as a CSV file.

---

## Sidebar Settings

| Setting                    | Default | Description                                                         |
| -------------------------- | ------- | ------------------------------------------------------------------- |
| Confidence threshold       | 0.25    | Lower = more detections; higher = fewer but more confident          |
| Enable price-axis mapping  | On      | Map pixel positions to real price values                            |
| Price axis minimum         | 0.0     | Lowest price visible on the Y-axis                                  |
| Price axis maximum         | 1000.0  | Highest price visible on the Y-axis                                 |
| Crop image before analysis | On      | Interactive crop before passing to model                            |
| Use full analysis pipeline | Off     | Runs TradingView preprocessor + OCR for auto price mapping (slower) |

---

## How OHLC is Extracted

1. **YOLO detection** — bounding box per candlestick, sorted left → right
2. **Color classification** — HSV green/red pixel count → bullish or bearish
3. **Body detection** — horizontal projection finds the candle body top/bottom
4. **Price mapping** — linear interpolation from pixel Y-coordinate using the provided price range:
   - **High** = top of bounding box
   - **Low** = bottom of bounding box
   - **Open / Close** = body bounds (order depends on direction)

---

## Model

| Property     | Value                                         |
| ------------ | --------------------------------------------- |
| Architecture | YOLOv8                                        |
| Weights file | `streamlit_app/best.pt`                       |
| Task         | Object detection (candlestick bounding boxes) |
| Input        | RGB chart image (any resolution)              |
| Classes      | `bullish_candle`, `bearish_candle`            |

---

## Troubleshooting

- **No candlesticks detected** — lower the confidence threshold in the sidebar (try 0.05–0.10).
- **Wrong prices** — make sure you enter the correct Y-axis min/max from the chart, or enable "Use full analysis pipeline" for automatic OCR-based mapping.
- **Model not found** — ensure `best.pt` is inside the `streamlit_app/` folder.
- **Import errors for `src`** — run the app from the project root, not from inside `streamlit_app/`.
# Candlestick-Chart-OHLC-Extractor-Streamlit-Web-App
