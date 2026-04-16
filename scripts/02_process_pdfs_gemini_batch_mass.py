from pydantic import BaseModel
from google.genai import errors
from _OCRBatchProcessor import *

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(
  filename='02_gemini_batch_mass.log', 
  filemode='a', 
  encoding='utf-8', 
  level=logging.WARNING) ## <=================== Change logging level here


class OCRLine(BaseModel):
    text: str
    x: int
    y: int
    width: int
    height: int
    confidence: float

class OCRResult(BaseModel):
    lines: list[OCRLine]

# ***************************** constants *****************************
SKIP_TEXT = "******* KEEPS FAILING! SKIPPING FOR NOW *******"
INITIAL_WAIT_SECONDS = 60 * 8 # 8 minutes
FOLLOWUP_WAIT_SECONDS = 60 * 1 # 1 minute
MAX_ITERATIONS = 1 # 1000
MAX_BATCHES_AT_ONCE = 100
MODEL_NAME ='gemini-flash-latest' # gemini-3-flash-preview <- is what this has been in 2/2026
MODEL_PROMPT = (
    "Your role is to perform OCR on the images you are prompted with. "
    "Extract all text from these images of columns from a medical directory. "
    "For each line of text, provide the text content, its bounding box coordinates "
    "in pixels, and a confidence score (between 0 and 1). "
    "Lines of text extend horizontally across the whole column. "
    "Return ONLY a JSON array with this exact format:\n"
    '[\n'
    '  {"text": "example text", "x": 10, "y": 20, "width": 100, "height": 15, "confidence": 0.95},\n'
    '  {"text": "wierd text; ṽ", "x": 10, "y": 30, "width": 90, "height": 19, "confidence": 0.50},\n'
    '  {"text": "more text", "x": 10, "y": 40, "width": 95, "height": 15, "confidence": 0.90}\n'
    ']\n'
    "Include all text, even small fragments. "
    "Coordinates should be exact, in pixels. "
    "Encode the bounding box in each line's x, y, width, and height fields. "
    "High confidence values close to 1 represent that that the text has been rendered correctly "
    "with little likelihood that the image contained different text. "
    "Lower confidence vales, as low as 0, represent uncertainty about what text the image contained. "
    "Include all symbols, punctuation, and line breaks, even if they look like noise. "
    "Encode special characters properly in JSON as UTF-8 characters, standardizing them across all images. "
    "For instance, there is a small dark cadeuceus at the end of some lines, occasionally followed by a G or N, encode that symbol as ▼. "
    "Places to expect ▼ include \"(l'08) ; ▼\", \"Prowell, James W. (b'73)-Mo.7,'96; (♁) ; ▼\", "
    "\"▼G\", \"'03; (l'03); 1444 N. 31st St.; U*; ▼G\", and \"Marx Bldg.; (A28); S*; ▼N\". "
    "Encode a sun cross or wheel cross that can appear between some right parentheses and dashes as ⊕. "
    "Places to expect ⊕ incude \"McCOLLUM, HERMAN E. (b'77)⊕-Mo.7,\", \"OLSON, EVALD (b'66)⊕-Kan.3,'07; (l'16)\", "
    "\"⊕-Mass.7,'06; Member Mass. Med. Soc.;\", and \"⊕-Pa.1,'00; (l'00); 491 E. State St.;\". "
    "Also, a diamond can appear after dashes and should be encoded as ◊. "
    "Places to expect ◊ include \"Bailey, Alexander Henry-◊; (l'89)\", "
    "\"Johnson, Granville Roswell (b 47) E-◊;\", and \"FREEMAN, JOHN F.-◊; 370 W. 10th St.;\". "
    "A triangle can appear after dashes and should be encoded as △, as in \"Veon, John E. (b'71)-△; (l 17); 2310 B.\" "
    "Stars can appear after one- or two-letter abbreviations and should be encoded as ★. "
    "Places to expect ★ include \"Bldg.; 2-4, 7-8; S★\", \"10-12, 3-5; I★\", "
    "\"Ridotto; 2-4; OALR★\", \"Op★\", and \"office, 314 W. 4th St.; (G1,3); R★\". "
    "† can appear in parentheses after an l, 1, or I, as in "
    "\"(l†)\", \"Harris, J. Monroe-◊; (l'†); not in practice\", "
    "\"Lindner, Carl W. (b'45)-O.9,'75; (l†); re-\", or \"'77, N.Y.5,'77; (1†) ; (A28) ; S\". "
    "♁ or ‡ can appear by themselves in parentheses, "
    "as in \"'89; (♁); S\", \"Stone, Robt. E.-Ga.5,'91; (♁); 648 Wood-\", "
    "\"Md.1,'82; (‡); 2927 St. Paul St.; 4-6\". or \"Murdock, Jos. L. (b'62)-Ga.10,'93; (‡)\". "
    "For punctuation like dashes, quotes, and apostrophes, use standard ASCII equivalents. "
    "Avoid encoding anything as non-ASCII characters beyond ▼, ⊕, ◊, ★, †, ♁, and ‡. "
    "Non-ASCII characters besides ▼, ⊕, ◊, ★, †, ♁, and ‡ decrease the confidence score. "
)

def create_batch_processor():
    return OCRBatchProcessor(
        logger, 
        MODEL_NAME, 
        MODEL_PROMPT, 
        "OCR line", 
        OCRLine, 
        OCRResult, 
        only_count_tokens=False,#True,
        max_batches_at_once=MAX_BATCHES_AT_ONCE,
        max_entries_per_batch=1,
        initial_wait_seconds=INITIAL_WAIT_SECONDS,
        followup_wait_seconds=FOLLOWUP_WAIT_SECONDS,
    )

if __name__ == "__main__":
    # 1. Setup Project Paths
    script_dir = Path(__file__).parent
    project_root = script_dir if (script_dir / "data").exists() else script_dir.parent
    preprocessed_dir = project_root / "data" / "01_preprocessed"
    metadata_path = preprocessed_dir / "all_metadata.csv"
    output_dir = project_root / "data" / "02_raw_batch_mass"
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_processor = None

    for i in range(MAX_ITERATIONS):
        try:
            print("*** Iteration", i, "***")
            if not batch_processor:
                batch_processor = create_batch_processor()
            batch_processor.batch_prompt(
                metadata_path, 
                output_dir, 
                "ocr_output_retest.jsonl",
                # [done_file], 
                # record_prompts_responses=True
            )
        except Exception as e:
            if isinstance(e, errors.APIError) and e.code == 429:
                print(f"*** main loop RESOURCE_EXHAUSTED exception, pausing for {INITIAL_WAIT_SECONDS/60} at {datetime.datetime.now()}... ***")
                logger.error(f"*** main loop RESOURCE_EXHAUSTED exception, pausing for {INITIAL_WAIT_SECONDS/60} at {datetime.datetime.now()}... ***")
                time.sleep(INITIAL_WAIT_SECONDS)
            else:
                print("*** main loop exception, pressing on ***")
                print(type(e).__name__, "-", e)
                # something went very wrong, scrub any ongoing batch jobs and processor
                for job in batch_processor.client.batches.list():
                    try:
                        batch_processor.client.batches.delete(name=job.name)
                    except:
                        pass
                try:
                    batch_processor = None
                except:
                    pass
                pass
