import random

## saving to PDF libs
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm


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