import re
import cv2
import pytesseract


def parse_exam_text(text):
    # מפריד לפי שאלות שמתחילות עם "1.", "2." וכו'
    raw_questions = re.split(r'\n(?=\d+\.)', text.strip())
    questions = []

    for raw in raw_questions:
        lines = raw.strip().split('\n')
        if not lines:
            continue
        question_line = lines[0]
        question_text = re.sub(r'^\d+\.\s*', '', question_line)
        options = [line.strip() for line in lines[1:] if line.strip()]
        questions.append({'question': question_text, 'options': options})

    return questions


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
