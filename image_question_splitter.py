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
DEBUG_DIR = 'mixed_questions//debug_answers'
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
        pad_top = 10 if idx == 0 else 0  # padding רק לתשובה א
        y0 = max(0, ans['y0'] - pad_top)
        y1 = answers[idx + 1]['y0'] if idx + 1 < len(answers) else h
        answer_imgs.append((ans['text'][0], img[y0:y1, :]))

    return question_img, answer_imgs


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

    # ערבב את הרשימה כולה
    shuffled = answers.copy()
    random.shuffle(shuffled)

    answer_imgs_only = [img for _, img in shuffled]

    merge_question_and_answers(q_img, answer_imgs_only, out_path)
    print(f"✅ Saved shuffled question: {out_path}")
    return out_path


def get_label_position(img):
    """
    Detects the position of a label (e.g., 'א.', 'ב.', '1.') in an RTL answer image.
    Returns (x, y, label_width) where:
    - x is the x-coordinate of the label's right edge
    - y is the vertical center
    - label_width is used to calculate alignment
    """
    import pytesseract
    import re
    import cv2

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    # Step 1: OCR on the top 60 pixels only
    top_strip = gray[:60, :]
    data = pytesseract.image_to_data(top_strip, lang='heb', config='--psm 6', output_type=pytesseract.Output.DICT)

    label_regex = r'^([א-ת]|[0-9])[.:\]]?$'  # Matches א 1. etc.

    for i, text in enumerate(data['text']):
        clean = text.strip()
        if re.fullmatch(label_regex, clean):
            x = data['left'][i]
            y = data['top'][i]
            w_box = data['width'][i]
            h_box = data['height'][i]
            return x + w_box, y + h_box // 2, w_box

    # Step 2: fallback via rightmost contour in top region
    _, thresh = cv2.threshold(gray[:60, :], 190, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # select contour closest to right edge
        c = min(contours, key=lambda cnt: cv2.boundingRect(cnt)[0])  # RTL = small x = right
        x, y, w_box, h_box = cv2.boundingRect(c)
        return x + w_box, y + h_box // 2, w_box

    # Step 3: fallback to top-right corner
    fallback_x = w + 60
    fallback_y = 25
    fallback_w = 30
    return fallback_x, fallback_y, fallback_w


def replace_label_by_first_pixel(img, new_label, font_scale=1.0, thickness=2, bin_thresh=400):
    """
    מוצא את הרכיב הוויזואלי הכי ימני שנראה כמו תווית תקנית (עברית), מוחק אותו ומצייר תווית חדשה.
    """
    from PIL import ImageFont, ImageDraw, Image as PILImage

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bw_inv = cv2.threshold(gray, bin_thresh, 255, cv2.THRESH_BINARY_INV)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bw_inv)

    h_img, w_img = gray.shape
    candidates = []

    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if w < 10 or h < 15:
            continue  # רעש קטן מדי
        if x > w_img * 0.6:
            roi = bw_inv[y:y + h, x:x + w]
            black = np.count_nonzero(roi)
            density = black / (w * h)
            candidates.append((x, y, w, h, density, i))

    if not candidates:
        if DEBUG:
            print("[DEBUG] No valid label candidates found in right side of image.")
        return img

    # נבחר את הרכיב עם ה־x הכי גדול (הכי ימני), מבין המועמדים
    best = max(candidates, key=lambda t: t[0])  # לפי x

    x, y, w, h, density, idx = best

    # מחיקה מורחבת סביב התווית
    pad_x = int(w * 0.6)
    pad_y = int(h * 0.4)
    x0 = max(0, x - pad_x)
    y0 = max(0, y - pad_y)
    x1 = min(img.shape[1], x + w + pad_x)
    y1 = min(img.shape[0], y + h + pad_y)

    img_copy = img.copy()
    img_copy[y0:y1, x0:x1] = 255  # מחיקה

    pil = PILImage.fromarray(img_copy)
    draw = ImageDraw.Draw(pil)

    try:
        font = ImageFont.truetype("arial.ttf", int(h * 1.8))
    except:
        font = ImageFont.load_default()

    draw.text((x, y), new_label, fill=(0, 0, 0), font=font)

    if DEBUG:
        draw.rectangle([(x0, y0), (x1, y1)], outline=(255, 0, 0), width=2)
        draw.text((5, 5), "DEBUG: old label erased", fill=(0, 0, 255))
        print(f"[DEBUG] Picked label @ x={x}, y={y}, w={w}, h={h}, density={density:.2f}")

    return np.array(pil)


def merge_question_and_answers(question_img, answer_imgs, out_path):
    from PIL import ImageFont, ImageDraw, Image as PILImage

    labels = ['א', 'ב', 'ג', 'ד', 'ה', 'ו']
    cleaned = []
    label_x_positions = []

    if DEBUG:
        debug_dir = os.path.join(os.path.dirname(out_path), "debug_answers")
        os.makedirs(debug_dir, exist_ok=True)

    for i, img in enumerate(answer_imgs):
        label = f"{labels[i]}."
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, bw_inv = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(bw_inv)

        h_img, w_img = gray.shape
        candidates = []

        for j in range(1, num_labels):
            x, y, w, h, area = stats[j]
            if w < 10 or h < 15:
                continue
            if x > w_img * 0.6:
                candidates.append((x, y, w, h))

        if not candidates:
            cleaned.append(img)
            label_x_positions.append(None)
            continue

        # הרכיב הימני ביותר בצד ימין
        x, y, w, h = max(candidates, key=lambda t: t[0])
        label_x_positions.append(x)

        # רקע מחוק באזור המקורי (להרחבה – בעתיד אפשר להסיר)
        img_copy = img.copy()

        pil = PILImage.fromarray(img_copy)
        draw = ImageDraw.Draw(pil)

        try:
            font = ImageFont.truetype("arial.ttf", int(h * 1.8))
        except:
            font = ImageFont.load_default()

        # קביעת X מנורמל
        draw_x = x
        if len([val for val in label_x_positions if val is not None]) >= 2:
            median_x = int(np.median([val for val in label_x_positions if val is not None]))
            if abs(x - median_x) > 10:
                extra_shift = 9  # כמה פיקסלים להזיז מעבר ליישור
                draw_x = median_x + extra_shift
                if DEBUG:
                    print(f"[DEBUG] Adjusted label '{label}' x from {x} → {draw_x} (with +{extra_shift}px extra shift)")

        # חישוב מיקום למחיקת הרקע
        bbox = font.getbbox(label)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        y_aligned = y

        # מחיקה במיקום החדש
        draw.rectangle(
            [(draw_x - 4, y_aligned - 2),
             (draw_x + text_w + 4, y_aligned + text_h + 2)],
            fill=(255, 255, 255)
        )

        # ציור האות המנורמלת
        draw.text((draw_x, y_aligned), label, fill=(0, 0, 0), font=font)

        # DEBUG
        if DEBUG:
            draw.rectangle(
                [(draw_x - 4, y_aligned - 2),
                 (draw_x + text_w + 4, y_aligned + text_h + 2)],
                outline=(255, 0, 0), width=2
            )
            draw.text((5, 5), "DEBUG: old label erased", fill=(0, 0, 255))

            q_base = os.path.splitext(os.path.basename(out_path))[0].replace('_mixed', '')
            debug_name = f"{q_base}_a{i}.png"
            debug_path = os.path.join(debug_dir, debug_name)
            cv2.imwrite(debug_path, np.array(pil))
            print(f"🖼️ Saved debug: {debug_path}")

        cleaned.append(np.array(pil))

    # מיזוג אנכי
    parts = [question_img] + cleaned
    total_h = sum(p.shape[0] for p in parts)
    max_w = max(p.shape[1] for p in parts)
    canvas = np.ones((total_h, max_w, 3), dtype=np.uint8) * 255

    y0 = 0
    for part in parts:
        h, w = part.shape[:2]
        canvas[y0:y0 + h, :w] = part
        y0 += h

    cv2.imwrite(out_path, canvas)


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
    clean_folder(DEBUG_DIR)

    process_page('input/Exam24BB-02.png', SPLIT_DIR, MIXED_DIR)
    process_page('input/Exam24BB-03.png', SPLIT_DIR, MIXED_DIR)
    process_page('input/Exam24BB-04.png', SPLIT_DIR, MIXED_DIR)
    # process_page('input/testing2.png', SPLIT_DIR, MIXED_DIR)
    # process_page('input/testing3.png', SPLIT_DIR, MIXED_DIR)
    # process_page('input/testing4.png', SPLIT_DIR, MIXED_DIR)
