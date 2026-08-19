import json
import pathlib
from typing import List

from bs4 import BeautifulSoup, Comment
from charset_normalizer import from_path
import re
import pandas as pd
from docling.document_converter import DocumentConverter, XBRLFormatOption
from pydantic import BaseModel, ConfigDict

# pd.read_html() for tables 


# Clean the html file 
content = str(from_path("../data/nvda-20260125.html").best())
soup = BeautifulSoup(content, 'html.parser')

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

# # Get all the divs
div_tags = soup.find_all("div")

class Chunk(BaseModel):
    item: str
    section: str
    subsection: str
    heading: str
    text: str
    # company: str
    # source: str # 10K or R-file,
    # document_content_hash: str

chunks: List[Chunk] = []

cover_page = True
current_item = ""
current_section = ""
current_subsection = ""
current_heading = ""
current_text = ""
for div in div_tags:

    # print(div.prettify())
    # Until you reach table of contents, skip everything
    # if (div.get_text().lower() == "table of contents"):
    #     cover_page = False
    #     div.decompose()

    # # Collect all info from now
    # if not cover_page:
        # print(div.prettify())
        # # Skip the Page of contents
        # if div["style"] == "min-height:42.75pt;width:100%": pass

        # Skip the line numbers

        # get the nested tags of the div
    children = div.contents
    for child_tag in children:
        # use onlt span tags
        if child_tag.name == "span":
            # print(1)
            try:
                style_attr = child_tag.attrs.get("style")
                weight_match = re.search("font-weight:(\\d+)", style_attr)
                style_match = re.search("font-style:(\\w+)", style_attr)
                size_match = re.search("font-size:(\\d+)", style_attr)
                font_weight:str|None = weight_match.group(1) if weight_match else None
                font_style:str|None = style_match.group(1) if style_match else None
                font_size:str|None = size_match.group(1) if size_match else None
                # print(font_weight, font_style, style_attr)

                # Get text
                span_text = child_tag.get_text()

                # Find an ITEM
                # print(div.prettify())
                # print(font_size, "\n")
                item = re.search("ITEM", span_text) or re.search("Item", span_text) 
                if item != None and (font_weight == "700" or font_size == "14"): 
                    if current_text != "" and current_item != "":
                        # print("done")
                        chunks.append(Chunk(
                            item=current_item, 
                            section=current_section, 
                            subsection=current_subsection, 
                            heading=current_heading, 
                            text=current_text))
                    current_item = span_text 
                    # If this changes, all below values should change asw
                    current_section = ""
                    current_subsection = ""
                    current_heading = ""
                    current_text = ""
                    # print("item: ", current_item)
                
                # Identify Headings
                elif (font_weight == "700" or font_size == "10") and font_style=="italic":
                    if current_text != "" and current_item != "":
                        # print("done")
                        chunks.append(Chunk(
                            item=current_item, 
                            section=current_section, 
                            subsection=current_subsection, 
                            heading=current_heading, 
                            text=current_text))
                    current_heading = span_text
                    current_text = ""
                    # print("current_heading", current_heading) 

                # Identify Sections
                elif font_weight == "700": 
                    if current_text != "" and current_item != "":
                        # print("done")
                        chunks.append(Chunk(
                            item=current_item, 
                            section=current_section, 
                            subsection=current_subsection, 
                            heading=current_heading, 
                            text=current_text))
                    current_section = span_text
                    current_subsection = ""
                    current_heading = ""
                    current_text = ""
                    # print("section: ", current_section)
                
                # Identify plain text
                elif font_weight == "400":
                    current_text = current_text + span_text + "\n"
                    # print("text:", current_text)

            except Exception as e:
                print("found error: ", e)
            # print(content.span.get_text())
            # Rule for Item

pretty = "\n\n".join(json.dumps(c.model_dump(), indent=2) for c in chunks)
pathlib.Path("chunks_debug.json").write_text(pretty)
            #    
            # Rule for section
            # Rule for subsection
            # Rule for text
            # Rule for bullet points

    
    
pretty = soup.prettify()
# text_only = soup.get_text(separator="\n")





# print(pretty)

# Notes about the document for AMD
# Everything in <div> <span> content </span> <div>, no nested sub structure
# Font weight distinguishes headers
    # HEADING like ITEM 1A. RISK FACTORS
#      <div style="margin-top:9pt">
#    <span style="color:#000000;font-family:'Arial',sans-serif;font-size:10pt;font-weight:700;line-height:120%">
#     ITEM 1A.       RISK FACTORS
#    </span>
#   </div>

    # section:
    # <span style="color:#000000;font-family:'Arial',sans-serif;font-size:10pt;font-weight:700;line-height:120%">
    # sometimes you see to of these back to back, in which case, the former is a heading and the next ones are subheadings

    # Subsection
    # <span style="color:#000000;font-family:'Arial',sans-serif;font-size:10pt;font-style:italic;font-weight:700;line-height:120%">

    # plain text:
    # <span style="color:#000000;font-family:'Arial',sans-serif;font-size:10pt;font-weight:400;line-height:120%">

    # page number:
#     <div style="height:42.75pt;position:relative;width:100%">
#    <div style="bottom:0;position:absolute;width:100%">
#     <div style="text-align:center">
#      <span style="color:#000000;font-family:'Arial',sans-serif;font-size:10pt;font-weight:400;line-height:120%">
#       5
#      </span>
#     </div>
#    </div>
#   </div>
#   <hr style="page-break-after:always"/>

    # Table of contents page link
#     <div style="min-height:42.75pt;width:100%">
#    <div>
#     <span style="color:#0000ff;font-family:'Arial',sans-serif;font-size:9pt;font-weight:400;line-height:120%;text-decoration:underline">
#      <a href="#i597c59c6d5f1435f9e98177202b657fc_7" style="color:#0000ff;font-family:'Arial',sans-serif;font-size:9pt;font-weight:400;line-height:120%;text-decoration:underline">
#       Table of Conten
#      </a>
#      <a href="#i597c59c6d5f1435f9e98177202b657fc_7" style="color:#0000ff;font-family:'Arial',sans-serif;font-size:9pt;font-weight:400;line-height:120%;text-decoration:underline">
#       t
#      </a>
#      <a href="#i597c59c6d5f1435f9e98177202b657fc_7" style="color:#0000ff;font-family:'Arial',sans-serif;font-size:9pt;font-weight:400;line-height:120%;text-decoration:underline">
#       s
#      </a>
#     </span>
#    </div>

# get rid of the tables from the report
# tables = soup.find_all('table')


# Get the nonnumeric and nonfraction tags
# Get the context tags
# Group everything by their specified context



# nonnumeric_tags = soup.find_all('ix:nonnumeric')
# nonfraction_tags = soup.find_all("ix:nonfraction")
# fraction_tags = soup.find_all("ix:fraction")
# hidden_tags = soup.find_all("ix:hidden")
# context_tags = soup.find_all("xbrli:context")

# for i in nonnumeric_tags:
#     print(i, "\n")

# for i in context_tags:
#     print(i)

#