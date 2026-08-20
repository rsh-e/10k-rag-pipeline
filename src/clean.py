from json import dumps
import json
from pathlib import Path
import pathlib
import re
from bs4 import BeautifulSoup
from typing import List
from charset_normalizer import from_path
from pydantic import BaseModel

# regex is compiled once
weight_match_re = re.compile("font-weight:(\\d+)") 
style_match_re = re.compile("font-style:(\\w+)") 
size_match_re = re.compile("font-size:(\\d+)")

class Citation(BaseModel):
    item: str
    section: str
    heading: str
    text: str
    company: str
    year: str
    source: str # 10K or R-file,
    file_path: str

class Properties:
    def __init__(self, style_attr):
        self.style_attr = style_attr
        self.font_weight = self.get_font_weight()
        self.font_style = self.get_font_style()
        self.font_size = self.get_font_size() 
        # self.has_item = self.get_has_item()

    def get_font_weight(self) -> str|None:
        weight_match = weight_match_re.search(self.style_attr)
        font_weight: str|None = weight_match.group(1) if weight_match else None
        return font_weight
    
    def get_font_style(self):
        style_match = style_match_re.search(self.style_attr)
        font_style = style_match.group(1) if style_match else None
        return font_style

    def get_font_size(self) -> str|None:
        size_match = size_match_re.search(self.style_attr)
        font_size: str|None = size_match.group(1) if size_match else None
        return font_size

    # def get_has_item(self) -> bool:
    #     item = re.search("ITEM", span_text) or re.search("Item", span_text) 
    #     if item != None: return True
    #     else: return False

def strip_file(soup: BeautifulSoup) -> None:
    # get rid of all the tables
    tables = soup.find_all("table")
    for table in tables:
        table.decompose()

    # get rid of the xblr tags
    xblr_tag = soup.find("ix:header")
    xblr_tag.decompose()

    # get rid of all images
    img_tags = soup.find_all("img")
    for img in img_tags:
        img.decompose()


    # dont return anything, soup is mutable

def get_citations(path) -> List[Citation]:
    content = str(from_path(path).best())
    soup = BeautifulSoup(content, 'html.parser')
    strip_file(soup)

    citations: List[Citation] = []

    current_item: str = ""
    current_section: str  = ""
    current_heading: str = ""
    current_text: str = ""

    path_obj = Path(path)
    file_name = path_obj.stem
    split_file_name = file_name.split("-")
    company = split_file_name[0]
    year = split_file_name[1][0:4]
    file_path = path

    # Get all the divs
    div_tags = soup.find_all("div") 
    for div in div_tags:
        print("\n")
        children = div.contents
        for child_tag in children:
            if child_tag.name == "span":
                try:
                    style_attr = child_tag.attrs.get("style")
                    props = Properties(style_attr=style_attr)
                    span_text = child_tag.get_text()

                    IS_HEADING: bool = (props.font_weight == "700" or props.font_size == "10") and props.font_style == "italic"
                    IS_SECTION: bool = (props.font_weight == "700")
                    IS_PLAIN_TEXT: bool = (props.font_weight == "400")
                    VALID_CITATION: bool = (current_text != "" and current_item != "")
                    
                    item = re.search("ITEM", span_text) or re.search("Item", span_text) 
                    if item != None and (props.font_weight == "700" or props.font_size == "14"): 
                        if VALID_CITATION:
                            citations.append(Citation(
                                item=current_item, 
                                section=current_section, 
                                subsection=current_subsection, 
                                heading=current_heading, 
                                text=current_text,
                                company=company,
                                year=year,
                                source=file_name,
                                file_path=file_path))
                        current_item = span_text 
                        # If this changes, all below values should change asw
                        current_section = ""
                        current_subsection = ""
                        current_heading = ""
                        current_text = ""

                    elif IS_HEADING:
                        if VALID_CITATION:
                            citations.append(Citation(
                                item=current_item, 
                                section=current_section, 
                                subsection=current_subsection, 
                                heading=current_heading, 
                                text=current_text,
                                company=company,
                                year=year,
                                source=file_name,
                                file_path=file_path))
                        current_heading = span_text
                        current_text = ""

                    elif IS_SECTION:
                        if VALID_CITATION:
                        # print("done")
                            citations.append(Citation(
                                item=current_item, 
                                section=current_section, 
                                subsection=current_subsection, 
                                heading=current_heading, 
                                text=current_text,
                                company=company,
                                year=year,
                                source=file_name,
                                file_path=file_path))
                        current_section = span_text
                        current_subsection = ""
                        current_heading = ""
                        current_text = ""    

                    elif IS_PLAIN_TEXT:
                        current_text = current_text + span_text + "\n"

                except Exception as e:
                    print("found error:", e)

    return citations

if __name__ == "__main__":
    path = "../data/ko-20251231.html"
    citations = get_citations(path)

    output_file_name = "NEW_chunks_debug.json"
    pretty = "\n\n".join(json.dumps(c.model_dump(), indent=2) for c in citations)
    pathlib.Path(output_file_name).write_text(pretty)
    print("Citations can be viewed on,", output_file_name)
