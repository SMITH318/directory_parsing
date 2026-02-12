import layoutparser as lp
import cv2
from pdf2image import convert_from_path
from pathlib import Path

def process_single_page(pdf_path, page_num, dpi=300):
    """Convert a single page from PDF to image."""
    try:
        images = convert_from_path(
            str(pdf_path), 
            dpi=dpi,
            first_page=page_num,
            last_page=page_num
        )
        if images:
            return images[0]
    except Exception as e:
        print(f"    Error loading page {page_num}: {e}")
    return None

def main():
    ################### Constant/paths start here ##########################
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    pdf_dir = project_root / "pdfs" #/ "problems"
    output_base = project_root / "data" / "01_preprocessed"
    output_base.mkdir(parents=True, exist_ok=True)
    
    # image = process_single_page(pdf_dir / "Alabama.pdf", 1)
    image = cv2.imread(pdf_dir / "Alabama_Page_01.png")
    image = image[..., ::-1]

    model = lp.EfficientDetLayoutModel('lp://efficientdet/PubLayNet/tf_efficientdet_d1/config',
                                 extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8],
                                 label_map={0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"})
    layout = model.detect(image)
    lp.draw_box(image, layout, box_width=3)


if __name__ == "__main__":
    main()