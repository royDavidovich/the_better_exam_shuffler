import random
import time
import cv2
import pytesseract
import numpy as np
import os

from exam_parser import split_question_and_answers
from utils import clean_folder, find_content_blocks

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
INPUT_DIR = 'input'
SPLIT_DIR = 'split_questions'
MIXED_DIR = 'mixed_questions'
DEBUG_DIR = 'mixed_questions//debug_answers'
DEBUG = False  # Toggle visual debug mode


def split_by_content(img_path, out_dir='split_questions'):
    # clean_folder(out_dir)
    img = cv2.imread(img_path)
    base = os.path.splitext(os.path.basename(img_path))[0]
    blocks = find_content_blocks(
        img,
        row_thresh_frac=0.05,
        min_gap=80,
        min_block_height=300
    )

    out_paths = []

    if not blocks:
        # No blocks detected – save the entire image as one block
        fname = f"{base}_q00.png"
        out_path = os.path.join(out_dir, fname)
        cv2.imwrite(out_path, img)
        out_paths.append(out_path)
        return out_paths

    for i, (y0, y1) in enumerate(blocks, 1):
        crop = img[y0:y1, :]
        fname = f"{os.path.splitext(os.path.basename(img_path))[0]}_q{i:02d}.png"
        out_path = os.path.join(out_dir, fname)
        cv2.imwrite(out_path, crop)
        out_paths.append(out_path)

    return out_paths


# --- OCR-based splitting & shuffling ---------------------------------------


def shuffle_answers_in_image(img_path, out_dir='mixed_questions'):
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(img_path))[0]
    q_img, answers = split_question_and_answers(img_path)
    out_path = os.path.join(out_dir, base + '_mixed.png')

    if answers is None:
        print(f"⚠️ No answers detected; {out_path}")
        original = cv2.imread(img_path)
        cv2.imwrite(out_path, original)
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
        if i >= len(labels):
            break
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
            if DEBUG:
                print(f"[DEBUG] No label candidate found for answer {label}. Forcing draw anyway.")

            img_copy = img.copy()
            pil = PILImage.fromarray(img_copy)
            draw = ImageDraw.Draw(pil)

            try:
                font = ImageFont.truetype("arial.ttf", int(img.shape[0] * 0.08))
            except:
                font = ImageFont.load_default()

            draw_x = 40
            draw_y = 40

            draw.text((draw_x, draw_y), label, fill=(0, 0, 0), font=font)

            if DEBUG:
                draw.rectangle(
                    [(draw_x - 4, draw_y - 2),
                     (draw_x + 60, draw_y + 25)],
                    outline=(255, 0, 0), width=2
                )
                draw.text((5, 5), "DEBUG: forced label", fill=(0, 0, 255))

                q_base = os.path.splitext(os.path.basename(out_path))[0].replace('_mixed', '')
                debug_name = f"{q_base}_a{i}_forced.png"
                debug_path = os.path.join(debug_dir, debug_name)
                cv2.imwrite(debug_path, np.array(pil))
                print(f"🖼️ Saved forced-label debug: {debug_path}")

            cleaned.append(np.array(pil))
            label_x_positions.append(draw_x)
            continue

        x, y, w, h = max(candidates, key=lambda t: t[0])
        label_x_positions.append(x)

        img_copy = img.copy()
        pil = PILImage.fromarray(img_copy)
        draw = ImageDraw.Draw(pil)

        try:
            font = ImageFont.truetype("arial.ttf", int(h * 1.8))
        except:
            font = ImageFont.load_default()

        draw_x = x
        if len([val for val in label_x_positions if val is not None]) >= 2:
            median_x = int(np.median([val for val in label_x_positions if val is not None]))
            if abs(x - median_x) > 10:
                draw_x = median_x + 9
                if DEBUG:
                    print(f"[DEBUG] Adjusted label '{label}' x from {x} → {draw_x}")

        bbox = font.getbbox(label)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        y_aligned = y

        draw.rectangle(
            [(draw_x - 4, y_aligned - 2),
             (draw_x + text_w + 4, y_aligned + text_h + 2)],
            fill=(255, 255, 255)
        )
        draw.text((draw_x, y_aligned), label, fill=(0, 0, 0), font=font)

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

    # unify widths before stacking
    parts = [question_img] + cleaned
    max_w = max(p.shape[1] for p in parts)

    for i in range(len(parts)):
        h, w = parts[i].shape[:2]
        if w < max_w:
            pad = np.ones((h, max_w - w, 3), dtype=np.uint8) * 255
            parts[i] = np.hstack([parts[i], pad])

    total_h = sum(p.shape[0] for p in parts)
    canvas = np.ones((total_h, max_w, 3), dtype=np.uint8) * 255

    y0 = 0
    for part in parts:
        h, w = part.shape[:2]
        canvas[y0:y0 + h, :w] = part
        y0 += h

    cv2.imwrite(out_path, canvas)


# --- Orchestration ---------------------------------------------------------

def process_page(img_path,
                 split_dir=SPLIT_DIR,
                 mixed_dir=MIXED_DIR):
    # if not DEBUG:
    #     clean_folder(split_dir)
    split_paths = split_by_content(img_path, split_dir)
    results = []
    for p in split_paths:
        out = shuffle_answers_in_image(p, mixed_dir)
        if out:  # Only include successfully shuffled ones
            results.append(out)
    return results


if __name__ == '__main__':
    clean_folder(SPLIT_DIR)
    clean_folder(MIXED_DIR)
    clean_folder(DEBUG_DIR)

    process_page('input/Exam24BB-02.png', SPLIT_DIR, MIXED_DIR)
    process_page('input/Exam24BB-03.png', SPLIT_DIR, MIXED_DIR)
    process_page('input/Exam24BB-04.png', SPLIT_DIR, MIXED_DIR)
    process_page('input/Exam24BB-05.png', SPLIT_DIR, MIXED_DIR)
    # process_page('input/testing2.png', SPLIT_DIR, MIXED_DIR)
    # process_page('input/testing3.png', SPLIT_DIR, MIXED_DIR)
    # process_page('input/testing4.png', SPLIT_DIR, MIXED_DIR)
