import streamlit as st
import cv2
import numpy as np
import fitz  # PyMuPDF
import io
import os
from PIL import Image
from ultralytics import YOLO

# ⚙️ Web Workspace Layout Initializer Configuration
st.set_page_config(page_title="Document Scanner Pro", page_icon="✂️", layout="centered")

st.title("✂️ Document Cropping ")
st.write("Bhavani Xerox")

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

    results = custom_ai_model(cv_image_input, verbose=False)
    doc_points = None

    for result in results:
        if result.obb is not None and len(result.obb.xyxyxyxy) > 0:
            raw_pts = result.obb.xyxyxyxy.cpu().numpy()  
            first_doc_pts = raw_pts[0]
            doc_points = first_doc_pts.reshape(4, 2)
            break

    if doc_points is None:
        h_pad, w_pad = int(h_orig * 0.04), int(w_orig * 0.04)
        cropped_fallback = orig_cv[h_pad:h_orig-h_pad, w_pad:w_orig-w_pad]
        return Image.fromarray(cv2.cvtColor(cropped_fallback, cv2.COLOR_BGR2RGB))

    rect = order_points_obb(doc_points).reshape(4, 2)

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

    x_zero, y_zero = 0, 0
    dst = np.array([
        [x_zero, y_zero],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")

    transform_matrix = cv2.getPerspectiveTransform(rect.astype(np.float32), dst)
    warped_result = cv2.warpPerspective(orig_cv, transform_matrix, (max_width, max_height), flags=cv2.INTER_CUBIC)

    return Image.fromarray(cv2.cvtColor(warped_result, cv2.COLOR_BGR2RGB))


# 📤 File Upload Tray Manager Panel
uploaded_files = st.file_uploader(
    "Upload image files or PDF documents here:", 
    type=["png", "jpg", "jpeg", "pdf","jfif"], 
    accept_multiple_files=True
)

if uploaded_files:
    # We use Streamlit session state to manage variables across button triggers cleanly
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
        st.session_state.output_name = ""
        st.session_state.mime_type = ""

    # Phase A: User clicks "Clip Crop" to process the files quietly in the background
    if st.session_state.processed_data is None:
        if st.button("✂️ Clip Crop", type="primary", use_container_width=True):
            with st.spinner("AI Processing files... Please wait..."):
                processed_images_cache = []
                input_is_pdf = False
                base_name = "scanned_output"

                for file in uploaded_files:
                    base_name = os.path.splitext(file.name)[0]
                    
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
                    else:
                        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
                        cv_img = cv2.imdecode(file_bytes, 1)
                        cropped_pil = crop_document_with_v4_ai_hd(cv_img)
                        processed_images_cache.append(cropped_pil)

                if len(processed_images_cache) > 0:
                    # ROUTE 1: Single image gets packaged back as an ultra-sharp JPG
                    if len(processed_images_cache) == 1 and not input_is_pdf:
                        img_buffer = io.BytesIO()
                        processed_images_cache[0].save(img_buffer, format="JPEG", quality=100, subsampling=0)
                        st.session_state.processed_data = img_buffer.getvalue()
                        st.session_state.output_name = f"perfect_crop_{base_name}.jpg"
                        st.session_state.mime_type = "image/jpeg"
                    
                    # ROUTE 2 & 3: Bulk photos or PDFs compile into a unified print PDF
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
                        
                        st.session_state.processed_data = pdf_output_buffer.getvalue()
                        st.session_state.output_name = f"cropped_bundle_{base_name}.pdf"
                        st.session_state.mime_type = "application/pdf"
            st.rerun()

    # Phase B: Once processing completes, show the native download button instantly
    else:
        st.success(f"🎉 Custom AI crop optimization complete!")
        st.download_button(
            label=f"✂️⬇️ Download Document ",
            data=st.session_state.processed_data,
            file_name=st.session_state.output_name,
            mime=st.session_state.mime_type,
            type="primary",
            use_container_width=True
        )
        
        # Simple reset button to allow scanning a new document batch
        if st.button("🔄 Scan Another Document", use_container_width=True):
            st.session_state.processed_data = None
            st.rerun()
