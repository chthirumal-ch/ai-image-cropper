import streamlit as st
import cv2
import numpy as np
import fitz  # PyMuPDF
import io
import os
from PIL import Image
from ultralytics import YOLO

# ⚙️ Web Workspace Layout Initializer Configuration
st.set_page_config(page_title="AI Document Scanner Pro", page_icon="📄", layout="wide")

st.title("📄 High-Fidelity Custom AI Document Scanner")
st.write("Upload raw smartphone capture images or PDF packets. The app will automatically crop the background surfaces out and deliver lossless prints.")

# 🧠 Check and load your private custom trained AI brain
MODEL_PATH = "best.pt"

@st.cache_resource
def load_custom_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Missing Core Model Weights: Could not locate '{MODEL_PATH}' in your repository root folder.")
        return None
    return YOLO(MODEL_PATH)

custom_ai_model = load_custom_model()

def order_points_obb(pts):
    """Consistent 4-point grouping allocation (top-left, top-right, bottom-right, bottom-left)"""
    pts = pts.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    x_sorted = pts[np.argsort(pts[:, 0]), :]
    left_most = x_sorted[:2, :]
    right_most = x_sorted[2:, :]

    tl = left_most[np.argmin(left_most[:, 1]), :]
    bl = left_most[np.argmax(left_most[:, 1]), :]
    tr = right_most[np.argmin(right_most[:, 1]), :]
    br = right_most[np.argmax(right_most[:, 1]), :]

    rect[0], rect[1], rect[2], rect[3] = tl, tr, br, bl
    return rect

def crop_document_with_v4_ai_hd(cv_image_input):
    """
    Tracks document edges using your custom AI brain, processes GrabCut boundary
    corrections, and returns crisp top-down array formats via high-fidelity interpolation.
    """
    if custom_ai_model is None:
        return Image.fromarray(cv2.cvtColor(cv_image_input, cv2.COLOR_BGR2RGB))
        
    orig_cv = cv_image_input.copy()
    h_orig, w_orig = cv_image_input.shape[:2]

    # Model inference calculations
    results = custom_ai_model(cv_image_input, verbose=False)
    doc_points = None

    for result in results:
        if result.obb is not None and len(result.obb.xyxyxyxy) > 0:
            raw_pts = result.obb.xyxyxyxy.cpu().numpy()  
            doc_points = raw_pts[0].reshape(4, 2)
            break

    # Guard Fallback
    if doc_points is None:
        h_pad, w_pad = int(h_orig * 0.04), int(w_orig * 0.04)
        cropped_fallback = orig_cv[h_pad:h_orig-h_pad, w_pad:w_orig-w_pad]
        return Image.fromarray(cv2.cvtColor(cropped_fallback, cv2.COLOR_BGR2RGB))

    rect = order_points_obb(doc_points).reshape(4, 2)

    # GrabCut Edge Fitting
    x_coords, y_coords = rect[:, 0], rect[:, 1]
    xmin, xmax = int(max(0, np.min(x_coords) - 15)), int(min(w_orig - 1, np.max(x_coords) + 15))
    ymin, ymax = int(max(0, np.min(y_coords) - 15)), int(min(h_orig - 1, np.max(y_coords) + 15))
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    mask = np.zeros(cv_image_input.shape[:2], np.uint8)

    rect_tuple = (xmin, ymin, xmax - xmin, ymax - ymin)
    cv2.grabCut(cv_image_input, mask, rect_tuple, bgdModel, fgdModel, 3, cv2.GC_INIT_WITH_RECT)

    refined_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    contours, _ = cv2.findContours(refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        largest_cnt = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(largest_cnt, True)
        approx = cv2.approxPolyDP(largest_cnt, 0.015 * perimeter, True)
        if len(approx) == 4:
            rect = order_points_obb(approx.reshape(4, 2))
        else:
            center = np.mean(rect, axis=0)
            for i in range(4):
                rect[i] = center + (rect[i] - center) * 1.03  

    rect = rect.reshape(4, 2)
    tl, tr, br, bl = rect[0], rect[1], rect[2], rect[3]

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = int(max(width_a, width_b))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = int(max(height_a, height_b))

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")

    transform_matrix = cv2.getPerspectiveTransform(rect.astype(np.float32), dst)
    # Applying High-Fidelity Cubic Restorations
    warped_result = cv2.warpPerspective(orig_cv, transform_matrix, (max_width, max_height), flags=cv2.INTER_CUBIC)

    return Image.fromarray(cv2.cvtColor(warped_result, cv2.COLOR_BGR2RGB))

# 📤 File Upload Tray Manager Panel
uploaded_files = st.file_uploader(
    "Drag & Drop your customer image files or PDF documents here:", 
    type=["png", "jpg", "jpeg", "pdf"], 
    accept_multiple_files=True
)

if uploaded_files:
    processed_images_cache = []
    input_is_pdf = False
    base_document_name = "scanned_output"

    for file in uploaded_files:
        base_document_name = os.path.splitext(file.name)[0]
        
        # Parse PDF Document page streams
        if file.name.lower().endswith(".pdf"):
            input_is_pdf = True
            pdf_bytes = file.read()
            pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
            
            for page_idx in range(len(pdf_document)):
                page = pdf_document[page_idx]
                pixmap = page.get_pixmap(dpi=150)
                image_data = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
                cv_img = cv2.cvtColor(np.array(image_data), cv2.COLOR_RGB2BGR)
                
                cropped_pil = crop_document_with_v4_ai_hd(cv_img)
                processed_images_cache.append(cropped_pil)
            pdf_document.close()

        # Parse standard incoming images
        else:
            file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
            cv_img = cv2.imdecode(file_bytes, 1)
            cropped_pil = crop_document_with_v4_ai_hd(cv_img)
            processed_images_cache.append(cropped_pil)

    # Display real-time output review preview layouts
    if len(processed_images_cache) > 0:
        st.subheader("🔍 Review Scan Output Panels")
        for idx, preview_page in enumerate(processed_images_cache):
            st.image(preview_page, caption=f"Cropped Page {idx + 1}", use_container_width=True)

        st.subheader("📦 Download Hub")
        # EXPORT AUTOMATIC ROUTE 1: Single image files download directly as a high-fidelity JPG image
        if len(processed_images_cache) == 1 and not input_is_pdf:
            img_buffer = io.BytesIO()
            # Forced quality=100 for zero text pixelation defects
            processed_images_cache[0].save(img_buffer, format="JPEG", quality=100, subsampling=0)
            
            st.download_button(
                label="📥 Download Clean Cropped Image (JPG)",
                data=img_buffer.getvalue(),
                file_name=f"perfect_crop_{base_document_name}.jpg",
                mime="image/jpeg",
                type="primary"
            )

        # EXPORT AUTOMATIC ROUTE 2 & 3: Bulk photos or multi-page documents get a single unified print PDF
        else:
            pdf_compiler = fitz.open()
            for pil_page in processed_images_cache:
                img_buffer = io.BytesIO()
                pil_page.save(img_buffer, format="JPEG", quality=98)
                img_buffer.seek(0)
                
                page_pdf_bytes = fitz.open("pdf", fitz.open(stream=img_buffer.getvalue(), filetype="jpeg").convert_to_pdf())
                pdf_compiler.insert_pdf(page_pdf_bytes)
                
            pdf_output_buffer = io.BytesIO()
            pdf_compiler.save(pdf_output_buffer)
            pdf_compiler.close()
            
            st.download_button(
                label="📥 Download All Pages Combined as One Print-Ready PDF",
                data=pdf_output_buffer.getvalue(),
                file_name=f"cropped_bundle_{base_document_name}.pdf",
                mime="application/pdf",
                type="primary"
            )
