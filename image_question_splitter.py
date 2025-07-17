import random
import shutil
import time

import cv2
import pytesseract
import re
import numpy as np
import os
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
INPUT_DIR = 'input'
SPLIT_DIR = 'split_questions'
MIXED_DIR = 'mixed_questions'


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


#
# def detect_question_starts(image_path):
#     """
#     Returns (question_starts, fallback_end, is_fallback, image)
#     """
#     img = cv2.imread(image_path)
#     rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     data = pytesseract.image_to_data(
#         rgb, lang='heb+eng', output_type=pytesseract.Output.DICT
#     )
#
#     entries = []
#     for i, txt in enumerate(data['text']):
#         t = txt.strip()
#         if not t:
#             continue
#         entries.append({
#             'text': t,
#             'y': data['top'][i],
#             'line': data['line_num'][i],
#             'block': data['block_num'][i]
#         })
#
#     # merge same-line entries
#     merged, cur, last = [], [], (-1, -1)
#     for e in entries:
#         key = (e['block'], e['line'])
#         if key != last and cur:
#             merged.append({
#                 'text': " ".join(x['text'] for x in cur),
#                 'y': min(x['y'] for x in cur)
#             })
#             cur = []
#         cur.append(e)
#         last = key
#     if cur:
#         merged.append({
#             'text': " ".join(x['text'] for x in cur),
#             'y': min(x['y'] for x in cur)
#         })
#
#     # detect headers
#     q_starts = [m['y'] for m in merged if re.search(r"מס[׳']?\s*\d+", m['text'])]
#     if q_starts:
#         print(f"✅ Detected {len(q_starts)} question(s) by header.")
#         return sorted(q_starts), None, False, img
#
#     # fallback by answers
#     answers = [m for m in merged if re.match(r"^[אבגדה]\.", m['text'])]
#     if answers:
#         ys = [a['y'] for a in answers]
#         y_min, y_max = min(ys), max(ys)
#         block_h = y_max - y_min
#         y0 = max(0, int(y_min - 3*block_h))
#         print("⚠️ Fallback mode: using answer block.")
#         print(f"↳ Answer block height={block_h}px, estimating start y={y0}")
#         return [y0], y_max, True, img
#
#     print("❌ No questions or answers detected.")
#     return [], None, False, img
#
# def crop_horizontal(img, pad_left=10, pad_right=10):
#     """
#     Crop left/right to first/last non-white pixel + padding.
#     """
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     _, thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
#     cols = np.any(thresh, axis=0)
#     if not cols.any():
#         return img
#     x_coords = np.where(cols)[0]
#     x0, x1 = x_coords[0], x_coords[-1] + 1
#     h, w = img.shape[:2]
#     x0 = max(0, x0 - pad_left)
#     x1 = min(w, x1 + pad_right)
#     return img[:, x0:x1]
#
# def split_questions(image_path, output_dir='output_questions'):
#     os.makedirs(output_dir, exist_ok=True)
#     base = os.path.splitext(os.path.basename(image_path))[0]
#
#     q_starts, fallback_end, is_fb, img = detect_question_starts(image_path)
#     img_h, img_w = img.shape[:2]
#     q_starts.sort()
#
#     if not q_starts:
#         print("❌ No question blocks to split.")
#         return
#
#     q_starts.append(img_h)
#     top_pad = 15
#     bottom_pad = 30
#
#     for i in range(len(q_starts)-1):
#         y0 = q_starts[i]
#         if not is_fb:
#             y0 = max(0, y0 - top_pad)
#         if is_fb and i == len(q_starts)-2 and fallback_end is not None:
#             y1 = min(img_h, fallback_end + bottom_pad)
#         else:
#             y1 = q_starts[i+1]
#
#         block = img[y0:y1, :]
#         block = crop_horizontal(block, pad_left=10, pad_right=20)
#
#         out_name = f"{base}_question_{i+1:02d}.png"
#         out_path = os.path.join(output_dir, out_name)
#         cv2.imwrite(out_path, block)
#         print(f"✅ Saved {out_name} (y={y0}-{y1}, crop width={block.shape[1]}px)")
#
# if __name__ == '__main__':
#     split_questions('input/testing3.png')

# --- Content-based splitting -----------------------------------------------

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


def split_by_content(img_path, out_dir='split_questions'):
    """
    Split a page image into question-block PNGs.
    Returns list of output paths.
    """
    clean_folder(out_dir)
    img = cv2.imread(img_path)
    base = os.path.splitext(os.path.basename(img_path))[0]
    blocks = find_content_blocks(img)

    out_paths = []
    for i, (y0, y1) in enumerate(blocks, 1):
        crop = img[y0:y1, :]
        fname = f"{base}_q{i:02d}.png"
        out_path = os.path.join(out_dir, fname)
        cv2.imwrite(out_path, crop)
        out_paths.append(out_path)
    return out_paths


# --- OCR-based splitting & shuffling ---------------------------------------
def get_label_position(img):
    """
    Detects the (x, y) position of an answer label in Hebrew (e.g., "א.") in the given image.
    Returns coordinates of the top-left corner of the label. If not found, uses a smart fallback near the top-right.
    """
    from pytesseract import image_to_data, Output
    import re

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    data = image_to_data(gray, lang='heb', output_type=Output.DICT)

    h, w = img.shape[:2]
    fallback_x = w - 60  # 60 pixels from right (RTL)
    fallback_y = 10

    for i, word in enumerate(data['text']):
        text = word.strip()
        # detect things like "א." or "ב." exactly
        if re.match(r"^[אבגדהו]\.", text):
            return data['left'][i], data['top'][i]

        # handle cases where "א" and "." are separate words, like: ["א", "."]
        if text in "אבגדהו" and i + 1 < len(data['text']):
            next_text = data['text'][i + 1].strip()
            if next_text == ".":
                return data['left'][i], data['top'][i]

    # fallback if no label was found
    return fallback_x, fallback_y


def split_question_and_answers(img_path):
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    data = pytesseract.image_to_data(
        rgb, lang='heb+eng', output_type=pytesseract.Output.DICT
    )

    entries = []
    for i, txt in enumerate(data['text']):
        txt = txt.strip()
        if not txt:
            continue
        entries.append({
            'text': txt,
            'y': data['top'][i],
            'h': data['height'][i],
            'line': data['line_num'][i],
            'block': data['block_num'][i]
        })

    # merge into lines
    merged = []
    cur, last_key = [], (-1, -1)
    for e in entries:
        key = (e['block'], e['line'])
        if key != last_key and cur:
            y0 = min(x['y'] for x in cur)
            y1 = max(x['y'] + x['h'] for x in cur)
            merged.append({'text': " ".join(x['text'] for x in cur), 'y0': y0, 'y1': y1})
            cur = []
        cur.append(e)
        last_key = key
    if cur:
        y0 = min(x['y'] for x in cur)
        y1 = max(x['y'] + x['h'] for x in cur)
        merged.append({'text': " ".join(x['text'] for x in cur), 'y0': y0, 'y1': y1})

    answers = [m for m in merged if re.match(r"^[אבגדהוז]\.", m['text'])]
    if not answers:
        return None, None

    answers.sort(key=lambda m: m['y0'])
    question_img = img[0:answers[0]['y0'], :]

    answer_imgs = []
    for idx, ans in enumerate(answers):
        y0 = ans['y0']
        y1 = answers[idx + 1]['y0'] if idx + 1 < len(answers) else h
        answer_imgs.append((ans['text'][0], img[y0:y1, :]))

    return question_img, answer_imgs


def merge_question_and_answers(question_img, answer_imgs, out_path):
    from PIL import ImageFont, ImageDraw, Image as PILImage

    labels = ['א', 'ב', 'ג', 'ד', 'ה', 'ו', 'ז', 'ח', 'ט', 'י']
    cleaned_answers = []

    for i, img in enumerate(answer_imgs):
        img_pil = PILImage.fromarray(img.copy())
        draw = ImageDraw.Draw(img_pil)

        try:
            font = ImageFont.truetype("arial.ttf", 28)
        except:
            font = ImageFont.load_default()

        # גלה איפה הייתה התווית המקורית
        x, y = get_label_position(img)

        # מחק את התווית הישנה (מלבן לבן קטן)
        cv2.rectangle(img, (x - 5, y - 5), (x + 60, y + 30), (255, 255, 255), -1)
        img_pil = PILImage.fromarray(img)

        # כתוב את התווית החדשה באותו מקום
        draw = ImageDraw.Draw(img_pil)
        draw.text((x, y), f"{labels[i]}.", fill=(0, 0, 0), font=font)

        cleaned_answers.append(np.array(img_pil))

    # מיזוג לשאלה אחת
    parts = [question_img] + cleaned_answers
    heights = [p.shape[0] for p in parts]
    max_w = max(p.shape[1] for p in parts)
    canvas = np.ones((sum(heights), max_w, 3), dtype=np.uint8) * 255
    y = 0
    for p in parts:
        h, w = p.shape[:2]
        canvas[y:y + h, :w] = p
        y += h

    cv2.imwrite(out_path, canvas)


def shuffle_answers_in_image(img_path, out_dir='mixed_questions'):
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(img_path))[0]
    q_img, answers = split_question_and_answers(img_path)
    out_path = os.path.join(out_dir, base + '_mixed.png')

    if answers is None:
        print(f"⚠️ No answers detected; {out_path}")
        return out_path

    # ערבוב עם seed
    seed = int(time.time_ns())
    random.seed(seed)
    print(f"🔀 Shuffling answers with seed {seed}")

    answer_imgs_only = [img for _, img in answers]  # התמונה בלבד
    random.shuffle(answer_imgs_only)

    # העבר לפונקציה את התמונות בלבד (ללא תווית ישנה)
    merge_question_and_answers(q_img, answer_imgs_only, out_path)
    print(f"✅ Saved shuffled question: {out_path}")
    return out_path


# --- Orchestration ---------------------------------------------------------

def process_page(img_path,
                 split_dir='split_questions',
                 mixed_dir='mixed_questions'):
    # 1) split page into question images
    split_paths = split_by_content(img_path, split_dir)
    # 2) shuffle each split image
    # clean_folder(mixed_dir)
    results = []
    for p in split_paths:
        out = shuffle_answers_in_image(p, mixed_dir)
        results.append(out)
    return results


if __name__ == '__main__':
    clean_folder(SPLIT_DIR)
    clean_folder(MIXED_DIR)

    # for fname in sorted(os.listdir(INPUT_DIR)):
    #     if not fname.lower().endswith('.png'):
    #         continue
    #     page = os.path.join(INPUT_DIR, fname)
    #     print(f"→ Processing {page}")
    #     split_paths = split_by_content(page, SPLIT_DIR)
    #     for p in split_paths:
    #         shuffle_answers_in_image(p, MIXED_DIR)

    # process_page('input/Exam24BB-02.png', 'split_questions', 'mixed_questions')
    # process_page('input/Exam24BB-03.png', 'split_questions', 'mixed_questions')
    # process_page('input/Exam24BB-04.png', 'split_questions', 'mixed_questions')
    process_page('input/Exam24BB-04.png', SPLIT_DIR, MIXED_DIR)
