from enum import unique
from fileinput import filename
from io import StringIO
from json import dumps
import json
import os
from pathlib import Path
import pathlib
import re
from bs4 import BeautifulSoup
from typing import List
from charset_normalizer import from_path
from langchain_text_splitters import markdown
import pandas as pd
import numpy as np
from pandas import DataFrame
from pydantic import BaseModel

# regex is compiled once
weight_match_re = re.compile("font-weight:(\\d+)") 
style_match_re = re.compile("font-style:(\\w+)") 
size_match_re = re.compile("font-size:(\\d+)")
SYMBOL_ONLY = re.compile(r"^[\$%—\-–\*†]*$")

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

def is_symbol_only(s: str) -> bool:
    return bool(re.match(r'^[\$%—\-–]*$', s.strip()))

# After spending 6 hours trying to merge 2 cells, I gave up and asked Claude, it did it in 5 seconds
def reconcile_columns(df: DataFrame) -> DataFrame:
    """Merge adjacent columns that are really one logical column split
    by colspan/ffill artifacts (bare '$' or duplicated values), while
    leaving genuinely distinct columns (e.g. different fiscal years)
    untouched even if a few of their cells happen to look mergeable."""
    cols = list(df.columns)
    result_cols = [cols[0]]
    merged = df[[cols[0]]].copy()

    for col in cols[1:]:
        prev_col = result_cols[-1]
        a = merged[prev_col].astype(str)
        b = df[col].astype(str)

        mergeable = True
        for av, bv in zip(a, b):
            av_s, bv_s = av.strip(), bv.strip()
            if av_s == bv_s or is_symbol_only(av_s) or is_symbol_only(bv_s) or av_s == "" or bv_s == "":
                continue
            mergeable = False
            break

        if mergeable:
            new_vals = []
            for av, bv in zip(a, b):
                av_s, bv_s = av.strip(), bv.strip()
                if av_s == bv_s:
                    new_vals.append(av_s)
                elif is_symbol_only(av_s) and av_s != "":
                    new_vals.append(f"{av_s}{bv_s}")
                elif is_symbol_only(bv_s) and bv_s != "":
                    new_vals.append(f"{av_s}{bv_s}")
                elif av_s == "":
                    new_vals.append(bv_s)
                else:
                    new_vals.append(av_s)
            merged[prev_col] = new_vals
        else:
            merged[col] = b
            result_cols.append(col)

    return merged[result_cols]

def drop_symbols(df: DataFrame) -> DataFrame:
    SYMBOL_ONLY = re.compile(r"^[\$%—\-–]*$") 
    cols_to_drop = []
    for col in df.columns:
        col_data = df[col].dropna().astype(str).str.strip()
        if col_data.empty or col_data.str.match(SYMBOL_ONLY).all():
            cols_to_drop.append(col)

    df = df.drop(columns=cols_to_drop)
    return df

def extract_tables(soup: BeautifulSoup) -> List[str]:
    tables = soup.find_all("table")
    tables_as_csv = []
    row_string = ""
    # pd.read_html(StringIO(tables))

    # Markdown approach
    markdown_tables = []
    for table in tables:
        try:
            table_df = pd.read_html(StringIO(str(table)))
        except:
            continue
        
        for df in table_df:
            
            df = df.dropna(axis=1, how="all")
            df = df.dropna(how="all")
            df = df.fillna("")
            df = reconcile_columns(df)
            # df = df.apply(lambda row: row.ffill(), axis=1)
            # df = df.T.drop_duplicates().T
            df = drop_symbols(df)
            unique_col = df.columns[0]
            unique_col_data = df[unique_col]
            no_duplicates = [unique_col]
            for col in df.columns[1:]:

                if not df[col].equals(unique_col_data):
                    # drop the column from the df
                    no_duplicates.append(col)
                    unique_col_data = df[col]
            
            df = df[no_duplicates]
            markdown_tables.append(df.to_markdown(index=0, tablefmt="grid"))


    for table in markdown_tables:
        print(table)
        
    # csv approach
    # for table in tables:    
    #     table_arr = []
    #     rows = table.contents
    #     for row in rows:
    #         data = row.contents
    #         for td in data:
    #             text: str|None = td.get_text()
    #             if text is not None and ("," in text):
    #                 text = '"'+ text +'"'
    #             if text == "$":
    #                 row_string = row_string + text
    #             else:
    #                 row_string = row_string + text + ","
    #         table_arr.append(row_string)
    #         row_string = ""
    #     tables_as_csv.append(table_arr)
    #     table_arr = []

    # for table in tables_as_csv:
    #     for row in table:
    #         print(row)
    #     print()

    # return tables_as_csv


def strip_file(soup: BeautifulSoup) -> None:
    # get rid of all the tables
    tables = soup.find_all("table")
    for table in tables:
        table.decompose()

    # get rid of the xblr tags
    xblr_tag = soup.find("ix:header")
    if xblr_tag is not None: xblr_tag.decompose()

    # get rid of all images
    img_tags = soup.find_all("img")
    for img in img_tags:
        img.decompose()


    # dont return anything, soup is mutable

def get_citations(path: str) -> List[Citation]:
    content = str(from_path(path).best())
    soup = BeautifulSoup(content, 'html.parser')
    print("hi")
    extract_tables(soup)
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
        children = div.contents
        for child_tag in children:
            if child_tag.name == "span":
                try:
                    style_attr = child_tag.attrs.get("style")
                    if style_attr is not None: 
                        props = Properties(style_attr=style_attr)
                    else:
                        continue
                    span_text = child_tag.get_text()

                    IS_HEADING: bool = (props.font_weight == "700" or props.font_size == "10") and props.font_style == "italic"
                    IS_SECTION: bool = (props.font_weight == "700" and span_text != "\u2022")
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
                    print(div.prettify())
                    print("found error:", e)

    # print(citations)
    return citations

from sqlalchemy import table
from transformers import AutoTokenizer
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("nomic-ai/nomic-embed-text-v1.5")

def token_count_histogram(file_name, citations: list[Citation]):
    counts = [len(tokenizer.encode(c.text)) for c in citations]
    counts = np.array(counts)

    print("-" * 100)
    print(file_name)
    print(f"n citations: {len(counts)}")
    print(f"min: {counts.min()}, max: {counts.max()}")
    print(f"mean: {counts.mean():.1f}, median: {np.median(counts):.1f}")
    print(f"p90: {np.percentile(counts, 90):.1f}, p95: {np.percentile(counts, 95):.1f}, p99: {np.percentile(counts, 99):.1f}")

    # simple bucketed histogram, no plotting deps needed
    buckets = [0, 100, 200, 400, 800, 1600, 3200, 5000, float("inf")]
    hist, _ = np.histogram(counts, bins=buckets)
    for i in range(len(hist)):
        lo, hi = buckets[i], buckets[i+1]
        label = f"{lo}-{hi}" if hi != float("inf") else f"{lo}+"
        print(f"{label:>12}: {hist[i]:>5}  {'#' * (hist[i] * 50 // max(hist.sum(), 1))}")

    return counts

if __name__ == "__main__":
    data_directory = "../data"
        
    path = "../data/cop-20251231.html"
    citations = get_citations(path)

    output_file_name = "NEW_chunks_debug.json"
    pretty = "\n\n".join(json.dumps(c.model_dump(), indent=2) for c in citations)
    pathlib.Path(output_file_name).write_text(pretty)
    print("Citations can be viewed on,", output_file_name)
