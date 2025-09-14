import re
import json
import numpy as np

from typing import Dict, List

from pypdf import PdfReader
import easyocr
from pdf2image import convert_from_path

HIERARCHY = ["livre", "titre", "chapitre", "section", "sous_section"]

class ParsingError(Exception):
    """Custom exception for parsing errors"""
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

def update_context(context, level, value):
    """
    Update parsing context and reset lower levels if needed.
    """
    context[level] = value
    idx = HIERARCHY.index(level)
    for lower in HIERARCHY[idx+1:]:
        context[lower] = ''
    return context

def parse(text: str, source_path: str, origine: str, article_pattern=1) -> List[Dict]:
    """
    Découpe le CESEDA en articles avec contexte hiérarchique et références.
    Hérite des infos (titre, chapitre, section, sous-section) de l’article précédent
    si elles ne sont pas trouvées explicitement.
    """
    # Regex hiérarchie
    livre_pattern = re.compile(r"(Livre\s+[IVXLC]+\s*:.*)")
    titre_pattern = re.compile(
        r"((?:Titre|TITRE)\s+"                                      # Titre or TITRE
        r"(?:[IVXLC]+|PREMIER|DEUXIÈME|TROISIÈME|QUATRIÈME|CINQUIÈME|SIXIÈME|SEPTIÈME|HUITIÈME|NEUVIÈME|DIXIÈME)"  # Roman numerals or French ordinals
        r"\s*:?.*)"
    )
    chapitre_pattern = re.compile(r"(Chapitre\s+.*)")
    section_pattern = re.compile(r"(Section\s+\d+\s*:.*)")
    sous_section_pattern = re.compile(r"(Sous-section\s+\d+\s*:.*)")
    
    # Regex article
    article_pattern_1 = re.compile(r"((?:Article|Art\.)\s+[A-Za-z\-]*\d*[-\dA-Za-z]*\s*:?)")
    article_pattern_2 = re.compile(r"^\s*(\d+)\.", re.MULTILINE)
    
    # Regex références internes
    reference_pattern = re.compile(r"articles?\s+(?:([LR])\.?\s*)?(\d+(?:-\d+)*)")

    translation_table = str.maketrans({
        " ": "",  # remove spaces
        "g": "9",
        "i": "1",
        "o": "0"
    })
        
    articles: Dict = {}
    
    # Découpage par articles
    article_pattern = article_pattern_1 if article_pattern == 1 else article_pattern_2
    parts = re.split(article_pattern, text)
    if len(parts) < 2:
        raise ParsingError("Aucun article trouvé dans le texte fourni.")
    
    for i in range(1, len(parts), 2):
        header = parts[i].strip()
        body = parts[i+1].strip() if i+1 < len(parts) else ""
        before = parts[i-1]  # texte avant l’article
        
        # Récupération du contexte de l’article précédent si existant
        prev = articles[list(articles.keys())[-1]] if articles else {}
        current_livre = prev.get("livre", "")
        current_titre = prev.get("titre", "")
        current_chapitre = prev.get("chapitre", "")
        current_section = prev.get("section", "")
        current_sous_section = prev.get("sous_section", "")

        current_dict = {
            "livre": current_livre,
            "titre": current_titre,
            "chapitre": current_chapitre,
            "section": current_section,
            "sous_section": current_sous_section
        }
        
        # Mise à jour seulement si trouvé dans "before" if we change section or chapitre or titre or livre we don't want to keep the previous one
        livre_match = livre_pattern.search(before)
        if livre_match:
            current_dict = update_context(current_dict, "livre", livre_match.group(1).strip())  

        titre_match = titre_pattern.search(before)
        if titre_match:
            current_dict = update_context(current_dict, "titre", titre_match.group(1).strip())
        
        chapitre_match = chapitre_pattern.search(before)
        if chapitre_match:
            current_dict = update_context(current_dict, "chapitre", chapitre_match.group(1).strip())
        
        section_match = section_pattern.search(before)
        if section_match:
            current_dict = update_context(current_dict, "section", section_match.group(1).strip())
        
        sous_section_match = sous_section_pattern.search(before)
        if sous_section_match:
            current_dict = update_context(current_dict, "sous_section", sous_section_match.group(1).strip())
        
        # Identifiant de l’article
        article_num = re.search(r"([LR]\.?\s*\d+[-\d]*)", header)
        art_code = article_num.group(1).translate(translation_table) if article_num else header
        code = art_code.split()[1] if ' ' in art_code else art_code
        code = code.translate(translation_table)
        art_code = f"Article {code}"
        
        # Références citées
        refs = [''.join(((m.group(1) or '').translate(translation_table), m.group(2))) for m in reference_pattern.finditer(body)]
        
        articles[art_code] = {
            **current_dict,
            "content": body,
            "referenced": refs,
            "source": source_path,
            "origine": origine
        }
    
    return articles

def get_page_text(page, header_lines=0, footer_lines=1):
    text = page.extract_text()
    lines = text.split("\n")
    content_lines = lines[:-footer_lines] if footer_lines > 0 else lines
    content_lines = content_lines[header_lines:] if header_lines > 0 else content_lines
    return "\n".join(content_lines)

def parse_pdf(file_dict: Dict, file_name: str) -> Dict[str, Dict]:
    """
    Parse a PDF file to extract and structure legal articles.
    file_dict should contain:
    - file_path: str, path to the PDF file
    - page_offset_start: int, optional, starting page index (default 0)
    - page_offset_end: int, optional, ending page index (default None, meaning till the end)
    - header_lines: int, optional, number of header lines to skip on each page (default 0)
    - footer_lines: int, optional, number of footer lines to skip on each page (default 0)
    Returns a dictionary of articles with their context and references.
    """
    file_path = file_dict["file_path"]
    page_offset_start = file_dict.get("page_offset_start", 0)
    page_offset_end = file_dict.get("page_offset_end", None)
    header_lines = file_dict.get("header_lines", 0)
    footer_lines = file_dict.get("footer_lines", 0)
    article_pattern = file_dict.get("article_pattern", 1)

    reader = PdfReader(file_path)
    full_text = ""
    for page in reader.pages[page_offset_start:page_offset_end]:
        full_text += get_page_text(page, header_lines, footer_lines) + "\n"
    
    articles = parse(full_text, source_path=file_path, origine=file_name, article_pattern=article_pattern)
    return articles

def extract_text_with_ocr(file_path: str) -> str:
    """
    Extract text from a PDF file using OCR.
    """
    pages = convert_from_path(file_path)
    ocr_reader = easyocr.Reader(['fr'], gpu=False)
    full_text = ""
    
    for page in pages:
        result_ocr = ocr_reader.readtext(np.array(page))
        for _, text, _ in result_ocr:
            full_text += text + "\n"
    return full_text

def parse_from_metadata(file_metadata_path: str) -> Dict[str, Dict]:
    """
    Parse multiple PDF files based on provided metadata.
    file_metadata is a list of dictionaries, each containing parameters for parse_pdf.
    Returns a combined dictionary of articles from all files.
    """
    file_metadata = json.load(open(file_metadata_path, 'r', encoding='utf-8'))
    for file_name, file_dict in file_metadata.items():
        try:
            print(f"Parsing {file_name}...")
            articles = parse_pdf(file_dict, file_name)
            # save articles to a json file
            json.dump(articles, open(f"sources/parsed/parsed_{file_name}.json", 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
            print(f"Parsed {len(articles)} articles from {file_name}.")
            print(f"Saved parsed articles to sources/parsed/parsed_{file_name}.json")
        except Exception as e:
            print(f"Error parsing {file_name}: {e} using OCR instead...")
            try:
                text = extract_text_with_ocr(file_dict["file_path"])
                articles = parse(text, source_path=file_dict["file_path"], origine=file_name, article_pattern=file_dict.get("article_pattern", 1))
                json.dump(articles, open(f"sources/parsed/parsed_{file_name}.json", 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
                print(f"Parsed {len(articles)} articles from {file_name} using OCR.")
                print(f"Saved parsed articles to sources/parsed/parsed_{file_name}.json")
            except Exception as e2:
                print(f"Error parsing {file_name} with OCR: {e2}")

########################################################## SUMMARIZATION ##########################################################

from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")


def summerize(text: str, summarizer = summarizer, max_length: int = 500) -> str:
    """
    Summarize the given text using a pre-trained transformer model.
    """
    # Summarize just the article
    summary_article = summarizer(text, max_length=max_length, min_length=30, do_sample=False)[0]['summary_text']

    return summary_article

def parsed_with_summaries(articles: Dict, max_length: int = 500) -> Dict:
    """
    Add summaries to each article in the parsed CESEDA
    3 levels of summarization
    input: article text + referenced articles + context (title, chapter, section, sub-section)
    output: 
        - summarized text for just the article (no context, no references)
        - summarized text for the article with context 
        - summarized text for the article with context and references
    
    """
    for art_code, art_data in articles.items():
        art_content = art_data["content"]
        art_summary = summerize(art_content, max_length=max_length)
        articles[art_code]["summary_level0"] = art_summary
        # Add context
        context = " ".join(filter(None, [art_data["livre"], art_data["titre"], art_data["chapitre"], art_data["section"], art_data["sous_section"]]))
        art_with_context = context + " " + art_content
        art_summary_context = summerize(art_with_context, max_length=max_length)
        articles[art_code]["summary_level1"] = art_summary_context
        # Add referenced articles
        referenced_texts = " ".join([articles[ref]["content"] for ref in art_data["referenced"] if ref in articles])
        art_with_references = art_with_context + " " + referenced_texts
        art_summary_references = summerize(art_with_references, max_length=max_length)
        articles[art_code]["summary_level2"] = art_summary_references
    return articles


if __name__ == "__main__":
    file_metadata_path = "sources/metadata.json"
    parse_from_metadata(file_metadata_path)
    # articles_with_summaries = parsed_with_summaries(articles, max_length=500)
    # with open("parsed_ceseda_with_summaries.json", "w", encoding="utf-8") as f:
    #     json.dump(articles_with_summaries, f, ensure_ascii=False, indent=4)