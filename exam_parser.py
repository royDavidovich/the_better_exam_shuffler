import re

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