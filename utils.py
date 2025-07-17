import os
import cv2
import numpy as np
from pdf2image import convert_from_path
from PIL import Image


def save_images_to_pdf(image_dir, output_pdf_path):
    """
    Convert all images in `image_dir` to a single PDF file.
    Images are sorted by filename.
    """
    image_files = sorted([
        f for f in os.listdir(image_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])

    if not image_files:
        print(f"❌ No images found in {image_dir}")
        return

    images = []
    for fname in image_files:
        img_path = os.path.join(image_dir, fname)
        img = Image.open(img_path).convert('RGB')
        images.append(img)

    first, rest = images[0], images[1:]
    os.makedirs(os.path.dirname(output_pdf_path), exist_ok=True)
    first.save(output_pdf_path, save_all=True, append_images=rest)
    print(f"📄 PDF created: {output_pdf_path}")


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
        thresh_val=240,
        morph_kernel=(3, 75),
        open_kernel=(3, 3),
        min_block_height=120,
        min_gap=50,
        pad_top=25,
        pad_bottom=25,
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

    return blocks
