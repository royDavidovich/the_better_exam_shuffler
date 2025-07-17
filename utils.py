import os
import random
import cv2
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm


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


def shuffle_array(arr):
    copy = arr[:]
    random.shuffle(copy)
    return copy


def shuffle_exam(questions):
    return [
        {
            'question': q['question'],
            'options': shuffle_array(q['options'])
        }
        for q in questions
    ]


def save_exam_to_pdf(questions, path):
    c = canvas.Canvas(path, pagesize=A4)
    width, height = A4
    margin = 2 * cm
    x = margin
    y = height - margin
    line_height = 16

    c.setFont("Helvetica", 12)

    for i, q in enumerate(questions, 1):
        question_line = f"{i}. {q['question']}"
        c.drawString(x, y, question_line)
        y -= line_height

        for j, opt in enumerate(q['options'], 1):
            option_line = f"{opt}"
            c.drawString(x + 20, y, option_line)
            y -= line_height

        y -= line_height // 2  # רווח נוסף בין שאלות

        # דף חדש אם הגענו לתחתית
        if y < margin:
            c.showPage()
            c.setFont("Helvetica", 12)
            y = height - margin

    c.save()


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
