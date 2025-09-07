import re
from typing import Dict, List
from pypdf import PdfReader

def parse_ceseda(text: str) -> Dict[str, Dict]:
    """
    Découpe le CESEDA en articles avec contexte hiérarchique et références.
    Hérite des infos (titre, chapitre, section, sous-section) de l’article précédent
    si elles ne sont pas trouvées explicitement.
    """
    # Regex hiérarchie
    livre_pattern = re.compile(r"(Livre\s+[IVXLC]+\s*:.*(?:\n.*){0,2})")
    titre_pattern = re.compile(r"(Titre\s+[IVXLC]+\s*:.*(?:\n.*){0,2})")
    chapitre_pattern = re.compile(r"(Chapitre\s+[IVXLC]+\s*:.*(?:\n.*){0,2})")
    section_pattern = re.compile(r"(Section\s+\d+\s*:.*(?:\n.*){0,2})")
    sous_section_pattern = re.compile(r"(Sous-section\s+\d+\s*:.*(?:\n.*){0,2})")
    
    # Regex article
    article_pattern = re.compile(r"(Article\s+[LR]\.?\s*\d+[-\d]*\s*:?)")
    
    # Regex références internes
    reference_pattern = re.compile(r"article\s+([LR]\.?\s*\d+[-\d]*)")
    
    articles: Dict = {}
    
    # Découpage par articles
    parts = re.split(article_pattern, text)
    
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
        
        # Mise à jour seulement si trouvé dans "before"
        livre_match = livre_pattern.search(before)
        if livre_match:
            current_livre = livre_match.group(1).strip()    

        titre_match = titre_pattern.search(before)
        if titre_match:
            current_titre = titre_match.group(1).strip()
        
        chapitre_match = chapitre_pattern.search(before)
        if chapitre_match:
            current_chapitre = chapitre_match.group(1).strip()
        
        section_match = section_pattern.search(before)
        if section_match:
            current_section = section_match.group(1).strip()
        
        sous_section_match = sous_section_pattern.search(before)
        if sous_section_match:
            current_sous_section = sous_section_match.group(1).strip()
        
        # Identifiant de l’article
        article_num = re.search(r"([LR]\.?\s*\d+[-\d]*)", header)
        art_code = article_num.group(1).replace(" ", "") if article_num else header
        
        # Références citées
        refs = [m.group(1).replace(" ", "") for m in reference_pattern.finditer(body)]
        
        articles[art_code] = {
            "livre": current_livre,
            "titre": current_titre,
            "chapitre": current_chapitre,
            "section": current_section,
            "sous_section": current_sous_section,
            "content": body,
            "referenced": refs
        }
    
    return articles

def get_page_text(page):
    """Extract text from a PDF page, excluding headers/footers if needed."""
    text = page.extract_text()
    lines = text.split("\n")
    content_lines = lines[:-1]
    return "\n".join(content_lines)

def parse_pdf(file_path: str) -> Dict[str, Dict]:
    reader = PdfReader(file_path)
    full_text = ""
    for page in reader.pages:
        full_text += get_page_text(page) + "\n"
    
    articles = parse_ceseda(full_text)
    return articles

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

