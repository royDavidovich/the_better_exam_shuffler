import os

from utils import convert_pdf_to_images, save_images_to_pdf_grid_high_quality
from image_question_splitter import process_page, clean_folder

INPUT_DIR = 'input'
OUTPUT_DIR = 'output'
TEMP_DIR = 'input_pdf_images'
SPLIT_DIR = 'split_questions'
MIXED_DIR = 'mixed_questions'
DEBUG_DIR = 'mixed_questions//debug_answers'


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pdf_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith('.pdf')]
    if not pdf_files:
        print("❌ No PDF found in 'input/' directory.")
        return

    input_pdf_path = os.path.join(INPUT_DIR, pdf_files[0])
    base_name = os.path.splitext(os.path.basename(input_pdf_path))[0]

    print(f"📥 Found PDF: {input_pdf_path}")

    clean_folder(TEMP_DIR)
    clean_folder(SPLIT_DIR)
    clean_folder(MIXED_DIR)
    clean_folder(DEBUG_DIR)

    image_paths = convert_pdf_to_images(input_pdf_path, TEMP_DIR)
    print(f"📄 Converted PDF to {len(image_paths)} image(s).")

    clean_folder(MIXED_DIR)

    for img_path in image_paths:
        print(f"🧪 Processing page: {img_path}")
        process_page(img_path)

    output_pdf_path = os.path.join(OUTPUT_DIR, base_name + "_shuffled.pdf")
    save_images_to_pdf_grid_high_quality(MIXED_DIR, output_pdf_path)

    print(f"✅ Done! PDF saved to: {output_pdf_path}")


if __name__ == '__main__':
    main()
