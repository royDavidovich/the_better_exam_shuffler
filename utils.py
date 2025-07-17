import os
import cv2
import numpy as np
from pdf2image import convert_from_path
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm

DEBUG = False  # Toggle visual debug mode


def clean_folder(path):
    """Delete all files in `path`. Creates it if it doesn't exist."""
    if os.path.isdir(path):
        # remove everything inside
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
    else:
        os.makedirs(path)


def convert_pdf_to_images(pdf_path, output_dir, dpi=300):
    poppler_dir = r".\poppler\Library\bin"

    os.makedirs(output_dir, exist_ok=True)
    images = convert_from_path(pdf_path, dpi=dpi, poppler_path=poppler_dir)

    saved_paths = []
    base = os.path.splitext(os.path.basename(pdf_path))[0]
    for i, img in enumerate(images):
        fname = f"{base}_page{i+1:02d}.png"
        out_path = os.path.join(output_dir, fname)
        img.save(out_path)
        saved_paths.append(out_path)

    return saved_paths


def find_content_blocks(
        img,
        thresh_val=235,
        morph_kernel=(3, 75),
        open_kernel=(3, 3),
        min_block_height=150,
        min_gap=50,
        pad_top=30,
        pad_bottom=30,
        row_thresh_frac=0.05
):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, morph_kernel)
    closed = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k_close)
    k_open = cv2.getStructuringElement(cv2.MORPH_RECT, open_kernel)
    clean = cv2.morphologyEx(closed, cv2.MORPH_OPEN, k_open)
    row_sum = np.sum(clean > 0, axis=1)
    row_thresh = int(row_sum.max() * row_thresh_frac)

    blocks = []
    in_block = False
    start = gap = 0

    for y, cnt in enumerate(row_sum):
        if cnt > row_thresh:
            if not in_block:
                in_block, start = True, y
            gap = 0
        else:
            if in_block:
                gap += 1
                if gap >= min_gap:
                    end = y - gap + 1
                    h = end - start
                    if h >= min_block_height:
                        y0 = max(0, start - pad_top)
                        y1 = min(img.shape[0], end + pad_bottom)
                        blocks.append((y0, y1))
                    in_block = False
                    gap = 0

    if in_block:
        end = len(row_sum)
        h = end - start
        if h >= min_block_height:
            y0 = max(0, start - pad_top)
            y1 = min(img.shape[0], end + pad_bottom)
            blocks.append((y0, y1))

    # DEBUG visualization
    if 'DEBUG' in globals() and DEBUG:
        dbg_img = img.copy()
        for y0, y1 in blocks:
            cv2.rectangle(dbg_img, (0, y0), (dbg_img.shape[1], y1), (0, 0, 255), 2)
        os.makedirs("debug_blocks", exist_ok=True)
        dbg_path = os.path.join("debug_blocks", f"block{y}_preview.png")
        cv2.imwrite(dbg_path, dbg_img)
        print(f"[DEBUG] Saved block preview: {dbg_path}")
        print(f"[DEBUG] Detected {len(blocks)} content blocks at:")
        for idx, (y0, y1) in enumerate(blocks):
            print(f"  Block {idx+1}: y0={y0}, y1={y1}, height={y1 - y0}")

    return blocks


def save_images_to_pdf_grid_high_quality(image_dir, output_pdf_path, margin_cm=1.5, spacing_px=20, assumed_dpi=300):
    image_files = sorted([
        f for f in os.listdir(image_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])
    if not image_files:
        print(f"❌ No images found in {image_dir}")
        return

    c = canvas.Canvas(output_pdf_path, pagesize=A4)
    page_width, page_height = A4
    margin = margin_cm * cm
    y_cursor = page_height - margin  # start from top
    max_width = page_width - 2 * margin
    min_margin_y = margin

    for fname in image_files:
        img_path = os.path.join(image_dir, fname)
        img = Image.open(img_path)
        img_width_px, img_height_px = img.size

        # Convert pixel size to points using assumed DPI
        img_width_pt = img_width_px * 72 / assumed_dpi
        img_height_pt = img_height_px * 72 / assumed_dpi

        # Scale image down if it's wider than allowed page width
        if img_width_pt > max_width:
            scale = max_width / img_width_pt
            img_width_pt *= scale
            img_height_pt *= scale

        # If not enough space on page for this image, move to new page
        if y_cursor - img_height_pt < min_margin_y:
            c.showPage()
            y_cursor = page_height - margin

        c.drawInlineImage(img_path, margin, y_cursor - img_height_pt, width=img_width_pt, height=img_height_pt)
        y_cursor -= img_height_pt + spacing_px

    c.save()
    print(f"📄 PDF generated (high quality): {output_pdf_path}")

