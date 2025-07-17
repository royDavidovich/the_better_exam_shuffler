# The Better Exam Shuffler 📄🎲  
*A smarter way to shuffle image-based CS exam questions.*

---

## 📝 Summary

This project was born out of frustration.  
Existing exam questions shufflers only handle plain text, while CS exams are full of **formulas**, **symbols**, and **images**.

So... I built a better one.

---

## 🚀 What It Does

- Loads scanned exams (PDF)
- Splits each page into question blocks
- Separates the **question** from its **visual answers**
- Detects Hebrew answer labels (א', ב', ג', ...) using layout-based logic (not just OCR)
- Randomly shuffles the answers
- Redraws the correct labels in consistent positions
- Saves the shuffled result as a clean, printable PDF

---

## 🔧 Tech Stack

- Python
- OpenCV + NumPy
- Tesseract (used lightly)
- Pillow (PIL) for drawing labels
- `pdf2image`, `reportlab` for PDF I/O
- A solid DEBUG mode to trace visual detection

---

## 📂 How To Use

1. Place your scanned exam PDF in the `input/` folder  
2. Run `main.py`  
3. Get a final shuffled PDF in the `output/` folder
---
