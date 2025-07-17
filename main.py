import os
from exam_parser import parse_exam_text
from utils import shuffle_exam, save_exam_to_pdf

INPUT_DIR = 'input'
OUTPUT_DIR = 'output'


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    txt_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.txt')]
    if not txt_files:
        print("❌ No .txt files found in 'input/' directory.")
        return

    for filename in txt_files:
        input_path = os.path.join(INPUT_DIR, filename)
        with open(input_path, encoding='utf-8') as f:
            text = f.read()

        print(f"🎯 Processing: {filename}")
        questions = parse_exam_text(text)
        mixed = shuffle_exam(questions)

        base_name = os.path.splitext(filename)[0]
        output_pdf = os.path.join(OUTPUT_DIR, f"{base_name}_mixed.pdf")
        save_exam_to_pdf(mixed, output_pdf)
        print(f"✅ Saved to: {output_pdf}\n")


if __name__ == '__main__':
    main()