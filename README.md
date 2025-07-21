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

## 🛠️ How to Build a Standalone EXE

To compile the project into a single executable file (`.exe`) for Windows, you can use **PyInstaller**. This bundles Python, your code, and the required dependencies (including the `tesseract` binaries) into one file.

### Steps:

1. Make sure you have PyInstaller installed:
    ```bash
    pip install pyinstaller
    ```

2. Make sure the `tesseract` folder (with `tesseract.exe` and `tessdata/`) is present **next to** `main.py` in your project directory. If you don't have `tesseract`, download and install it.

3. Run the following command from your project folder:
    ```bash
    pyinstaller --onefile --add-data "tesseract;tesseract" main.py
    ```

4. After the build completes, find your executable in the `dist/` folder, named `main.exe`.

5. Copy the `main.exe` to any Windows machine and run it without needing Python installed.

---

If you modify the project or update dependencies, re-run the build command to generate a new executable.

---