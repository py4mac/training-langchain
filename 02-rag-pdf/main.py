import base64
import io
import os
import re

import pymupdf
import requests
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from rapidocr import RapidOCR

load_dotenv()

pdf_to_parse = "https://www.hub-franceia.fr/wp-content/uploads/2025/02/Hub-France-IA-fevrier-2025-Feuillet-Agents-experts-en-IAGen.pdf"

# Moteur OCR initialisé une seule fois (le chargement des modèles est coûteux)
ocr_engine = RapidOCR()

# Modèle multimodal (vision) utilisé pour décrire les images, via OpenRouter
vision_model = init_chat_model(
    model="google/gemini-3.8-flash",
    model_provider="openai",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)

IMAGE_DESCRIPTION_PROMPT = (
    "Tu analyses une image extraite d'un document PDF afin d'alimenter un système RAG. "
    "Décris en français, de façon factuelle et concise, ce que montre l'image : "
    "type (photo, schéma, graphique, tableau, logo...), éléments principaux, "
    "relations entre eux et informations clés (chiffres, tendances, libellés). "
    "N'invente rien et ne recopie pas mot à mot le texte déjà lisible."
)

# Taille minimale (en points PDF, 1 pt = 1/72 pouce) de la zone affichée d'une image :
# évite logos et icônes
MIN_IMAGE_SIZE = 50
# Résolution du rendu des zones d'image (suffisante pour l'OCR et le modèle vision)
RENDER_DPI = 200


def ocr_image(image_bytes: bytes) -> str:
    """Extrait le texte d'une image via RapidOCR.
    Args:
        image_bytes: Le contenu binaire de l'image (PNG, JPEG, ...).
    Returns:
        Le texte reconnu dans l'image (chaîne vide si aucun texte).
    """
    result = ocr_engine(image_bytes)
    if not result.txts:
        return ""
    return " ".join(result.txts)


def describe_image(png_bytes: bytes) -> str:
    """Génère une description textuelle d'une image via un LLM multimodal.
    Args:
        png_bytes: Le contenu binaire de l'image au format PNG.
    Returns:
        La description de l'image (chaîne vide en cas d'erreur).
    """
    image_b64 = base64.b64encode(png_bytes).decode("utf-8")
    message = HumanMessage(
        content=[
            {"type": "text", "text": IMAGE_DESCRIPTION_PROMPT},
            {"type": "image", "base64": image_b64, "mime_type": "image/png"},
        ]
    )
    try:
        return " ".join(vision_model.invoke([message]).text.split())
    except Exception as e:
        print(f"ERROR description image : {e}")
        return ""


def extract_images_text(page: pymupdf.Page, with_description: bool = True) -> list[str]:
    """Extrait toutes les images d'une page et retourne leur texte (OCR) et leur description (LLM).
    Args:
        page: La page dont on veut traiter les images.
        with_description: Active la description des images par le modèle vision.
    Returns:
        Une liste de textes, un par image (description + texte OCR).
    """
    texts = []
    # On rend la zone de la page où l'image est affichée plutôt que d'extraire l'image brute :
    # le rendu applique masques de transparence (SMask), CMJN, rotations... Certaines images
    # (ex. p.5 du feuillet Hub France IA) ne sont qu'un aplat noir dont le dessin est dans le masque.
    seen_bboxes = set()
    for img_index, info in enumerate(page.get_image_info()):
        bbox = pymupdf.Rect(info["bbox"]) & page.rect
        if bbox.width < MIN_IMAGE_SIZE or bbox.height < MIN_IMAGE_SIZE:
            continue
        key = tuple(round(v) for v in bbox)
        if key in seen_bboxes:
            continue
        seen_bboxes.add(key)
        try:
            pix = page.get_pixmap(clip=bbox, dpi=RENDER_DPI)
            png_bytes = pix.tobytes("png")
            parts = []
            if with_description:
                description = describe_image(png_bytes)
                if description:
                    parts.append(f"Description : {description}")
            image_text = ocr_image(png_bytes)
            if image_text:
                parts.append(f"Texte : {image_text}")
            if parts:
                texts.append(" ".join(parts))
        except Exception as e:
            print(f"ERROR OCR page {page.number + 1}, image {img_index}: {e}")
    return texts


def read_pdf_content(pdf_url: str, with_images: bool = True, with_description: bool = True) -> str:
    """Télécharge et lit un fichier PDF à partir d'une URL, puis retourne son contenu sous forme de texte.
    Les images sont décrites par un LLM vision et leur texte est extrait par OCR.
    Args:
        pdf_url: L'URL du fichier PDF.
        with_images: Active le traitement des images (OCR + description).
        with_description: Active la description des images par le modèle vision.
    Returns:
        Une chaîne de caractères contenant le texte du PDF (et de ses images).
    """
    content = ""
    try:
        # Téléchargement du contenu du PDF depuis l'URL
        response = requests.get(pdf_url, timeout=60)
        response.raise_for_status()

        # Ouverture du PDF en mémoire avec PyMuPDF
        with pymupdf.open(stream=io.BytesIO(response.content), filetype="pdf") as pdf_document:
            for page in pdf_document:
                text = page.get_text()  # Récupère le texte brut au format UTF-8
                doc_txt = re.sub(r" {5,}", " <unknown> ", text.replace("\n", " "))
                content += doc_txt

                if with_images:
                    for image_text in extract_images_text(page, with_description):
                        content += f" [IMAGE p.{page.number + 1}] {image_text} [/IMAGE] "
    except Exception as e:
        print(f"ERROR in read_pdf_content function : {e}")

    return content


if __name__ == "__main__":
    # Test de la fonction avec l'URL définie précédemment
    pdf_text = read_pdf_content(pdf_to_parse)
    print(f"Longueur du texte extrait : {len(pdf_text)} caractères")
    print(pdf_text[:10000000] + "...")
