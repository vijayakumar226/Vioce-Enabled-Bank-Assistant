import io
import logging
import os
import platform
import re

import httpx
from PIL import Image, ImageEnhance, ImageFilter

try:
    import pytesseract
except ImportError:  # pragma: no cover - optional dependency at runtime
    pytesseract = None

logger = logging.getLogger(__name__)

if platform.system() == "Windows" and pytesseract is not None:
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def split_questions(extracted_text: str) -> list[str]:
    """
    Split extracted text into individual questions.
    Returns list of questions found.
    """
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", extracted_text)
    text = re.sub(r"([0-9])([A-Z])", r"\1 \2", text)

    questions = re.split(r"\?", text)

    cleaned: list[str] = []
    for question in questions:
        question = question.strip()
        question = re.sub(r"^[•\-\*\d\.\s]+", "", question)
        question = question.strip()
        words = question.split()
        if len(words) >= 4:
            cleaned.append(question + "?")

    return cleaned


async def extract_text_from_image(image_bytes: bytes) -> str:
    """
    Extract text from image using HF Inference API.
    Falls back to pytesseract if HF fails.
    """
    try:
        logger.info("Trying HF TrOCR model...")
        hf_token = os.environ.get("HF_TOKEN", "")
        if not hf_token:
            raise ValueError("No HF_TOKEN found")

        model = "microsoft/trocr-large-printed"
        url = f"https://api-inference.huggingface.co/models/{model}"
        headers = {
            "Authorization": f"Bearer {hf_token}",
            "Content-Type": "application/octet-stream",
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(url, headers=headers, content=image_bytes)

        logger.info("HF TrOCR status: %s", response.status_code)
        logger.info("HF TrOCR response: %s", response.text[:200])

        if response.status_code == 200:
            result = response.json()
            logger.info("HF TrOCR parsed: %s", result)

            if isinstance(result, list) and result:
                text = (result[0].get("generated_text") or "").strip()
                if text and len(text) > 3:
                    logger.info("HF TrOCR result: %s", text)
                    return text

            if isinstance(result, dict):
                text = (result.get("generated_text") or "").strip()
                if text and len(text) > 3:
                    logger.info("HF TrOCR result: %s", text)
                    return text
    except Exception as exc:
        logger.error("HF TrOCR error: %s", exc)

    try:
        logger.info("Trying Tesseract OCR...")
        if pytesseract is None:
            raise RuntimeError("pytesseract is not installed")

        image = Image.open(io.BytesIO(image_bytes))
        if image.mode != "RGB":
            image = image.convert("RGB")

        width, height = image.size
        if width < 1000:
            scale = 1000 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = image.resize((new_width, new_height), Image.LANCZOS)

        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2.0)
        image = image.convert("L")
        image = image.filter(ImageFilter.SHARPEN)

        custom_config = (
            r"--oem 3 --psm 6 "
            r"-c tessedit_char_whitelist="
            r"abcdefghijklmnopqrstuvwxyz"
            r"ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            r"0123456789 .,?!-:/@()"
        )

        text = pytesseract.image_to_string(image, config=custom_config).strip()
        logger.info("Tesseract result: '%s'", text)

        if text and len(text) > 3:
            return text
    except Exception as exc:
        logger.error("Tesseract failed: %s", exc)

    try:
        logger.info("Trying simple Tesseract OCR...")
        if pytesseract is None:
            raise RuntimeError("pytesseract is not installed")

        image = Image.open(io.BytesIO(image_bytes))
        text = pytesseract.image_to_string(image).strip()
        logger.info("Simple Tesseract result: '%s'", text)

        if text and len(text) > 3:
            return text
    except Exception as exc:
        logger.error("Simple Tesseract failed: %s", exc)

    return ""
