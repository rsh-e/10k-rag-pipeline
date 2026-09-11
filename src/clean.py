from io import StringIO
import json
from pathlib import Path
import pathlib
import re
from bs4 import BeautifulSoup
from typing import List
from charset_normalizer import from_path
import pandas as pd
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
    nearby_text: str = ""
    company: str
    year: str
    is_table: bool = False
    flatten_table: str = ""
    source: str  # 10K or R-file,
    file_path: str


class Properties:
    def __init__(self, style_attr):
        self.style_attr = style_attr
        self.font_weight = self.get_font_weight()
        self.font_style = self.get_font_style()
        self.font_size = self.get_font_size()
        # self.has_item = self.get_has_item()

    def get_font_weight(self) -> str | None:
        weight_match = weight_match_re.search(self.style_attr)
        font_weight: str | None = weight_match.group(1) if weight_match else None
        return font_weight

    def get_font_style(self):
        style_match = style_match_re.search(self.style_attr)
        font_style = style_match.group(1) if style_match else None
        return font_style

    def get_font_size(self) -> str | None:
        size_match = size_match_re.search(self.style_attr)
        font_size: str | None = size_match.group(1) if size_match else None
        return font_size


def is_symbol_only(s: str) -> bool:
    return bool(re.match(r"^[\$%—\-–]*$", s.strip()))


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
            if av_s == "" or bv_s == "" or is_symbol_only(av_s) or is_symbol_only(bv_s):
                continue
            if av_s == bv_s:
                continue
            a_num, b_num = _numeric_core(av_s), _numeric_core(bv_s)
            if a_num is not None and a_num == b_num:
                continue
            mergeable = False
            break

        if mergeable:
            new_vals = []
            for av, bv in zip(a, b):
                av_s, bv_s = av.strip(), bv.strip()
                if av_s == "":
                    new_vals.append(bv_s)
                elif bv_s == "":
                    new_vals.append(av_s)
                elif is_symbol_only(av_s):
                    new_vals.append(f"{av_s}{bv_s}")
                elif is_symbol_only(bv_s):
                    new_vals.append(f"{av_s}{bv_s}")
                elif av_s == bv_s:
                    new_vals.append(av_s)
                else:
                    new_vals.append(av_s if len(av_s) >= len(bv_s) else bv_s)
            merged[prev_col] = new_vals  # write the merged values back
        else:
            result_cols.append(col)  # keep this as its own column
            merged[col] = df[col].values

    return merged[result_cols]


def promote_first_row_as_header(df: DataFrame) -> DataFrame:
    """Assumes the header values all live in row 0 (or the first row with any
    non-empty content). Renames columns from that row, then drops it."""
    header_row = None
    for idx, row in df.iterrows():
        if row.astype(str).str.strip().replace({"nan": "", "None": ""}).ne("").any():
            header_row = idx
            break

    if header_row is None:
        return df

    new_names = {}
    for col in df.columns:
        val = str(df.loc[header_row, col]).strip()
        new_names[col] = val if val not in ("", "nan", "None") else str(col)

    df = df.rename(columns=new_names)
    df = df.drop(index=header_row).reset_index(drop=True)
    return df


def drop_symbols(df: DataFrame) -> DataFrame:
    SYMBOL_ONLY = re.compile(r"^[\$%—\-–]*$")
    cols_to_drop = []
    for col in df.columns:
        col_data = df[col].dropna().astype(str).str.strip()
        if col_data.empty or col_data.str.match(SYMBOL_ONLY).all():
            cols_to_drop.append(col)

    df = df.drop(columns=cols_to_drop)
    return df


def flatten_table(df: DataFrame) -> str:
    table_text = ""

    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    #     print(df)

    # Get the column names
    labels = df.columns.values.tolist()

    # pandas can only iterate through columns, to transpose
    df = df.transpose()
    columns = df.columns.values
    # y = []
    for col in columns:
        for i, data in enumerate(df[col]):
            if data == "——":
                data = ""
            table_text = table_text + labels[i] + ": " + data + ", "
        table_text = table_text + "\n"
    return table_text


def _numeric_core(s: str) -> str | None:
    """Strip $, commas, %, accounting-style parens off a value and return
    its canonical numeric string, so '$12,299', '12299', and '12299.0'
    all compare equal. Returns None if the string isn't numeric at all."""
    cleaned = s.strip().replace("$", "").replace(",", "").replace("%", "").strip()
    if cleaned == "":
        return None
    neg = cleaned.startswith("(") and cleaned.endswith(")")
    if neg:
        cleaned = cleaned[1:-1]
    try:
        f = float(cleaned)
        f = -f if neg else f
        return str(int(f)) if f == int(f) else str(f)
    except ValueError:
        return None


def _cols_equivalent(s1: pd.Series, s2: pd.Series) -> bool:
    if s1.equals(s2):
        return True
    for x, y in zip(s1.astype(str), s2.astype(str)):
        x, y = x.strip(), y.strip()
        if x == y or is_symbol_only(x) or is_symbol_only(y) or x == "" or y == "":
            continue
        if _numeric_core(x) is not None and _numeric_core(x) == _numeric_core(y):
            continue
        return False
    return True


def dedupe_repeated_row_values(df: DataFrame) -> DataFrame:
    """Some footnote/disclaimer rows originate from a single spanning cell
    (colspan) that pd.read_html duplicates into every column. When a row's
    non-empty values are all identical, keep the text once in the first
    column and blank the rest instead of repeating it once per column."""
    df = df.copy()
    for idx, row in df.iterrows():
        vals = [str(v).strip() for v in row]
        non_empty = [v for v in vals if v not in ("", "nan", "None")]
        if len(non_empty) > 1 and len(set(non_empty)) == 1:
            new_row = [""] * len(vals)
            new_row[0] = non_empty[0]
            df.loc[idx] = new_row
    return df


def extract_table(table) -> tuple[str, str] | None:
    try:
        table_df = pd.read_html(StringIO(str(table)))
    except ValueError:
        return None

    for df in table_df:
        df = df.dropna(axis=1, how="all")
        df = df.dropna(how="all")
        df = df.fillna("")
        df = reconcile_columns(df)
        df = drop_symbols(df)
        df = promote_first_row_as_header(df)

        # drop columns that are equivalent (same values, possibly different
        # formatting like "$12299" vs "12299" or "79826" vs "79826.0")
        keep_idx = [0]
        unique_col_data = df.iloc[:, 0]
        for i in range(1, len(df.columns)):
            col_data = df.iloc[:, i]
            if not _cols_equivalent(col_data, unique_col_data):
                keep_idx.append(i)
                unique_col_data = col_data
        df = df.iloc[:, keep_idx]

        df = dedupe_repeated_row_values(df)

        # To get rid of 'Table of Contents' and Page Numbers that are formatted as tables
        if len(df.columns) < 3 and len(df) < 2:
            return None

        # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        #     print(df)
        flattened_table = flatten_table(df)
        return flattened_table, df.to_markdown(index=0, tablefmt="grid")

    return None


def strip_file(soup: BeautifulSoup) -> None:
    # get rid of the xblr tags
    xblr_tag = soup.find("ix:header")
    if xblr_tag is not None:
        xblr_tag.decompose()

    # get rid of all images
    img_tags = soup.find_all("img")
    for img in img_tags:
        img.decompose()
    # dont return anything, soup is mutable


def get_citations(path: str) -> List[Citation]:
    content = str(from_path(path).best())
    soup = BeautifulSoup(content, "html.parser")
    strip_file(soup)

    citations: List[Citation] = []

    current_item: str = ""
    current_section: str = ""
    current_heading: str = ""
    current_text: str = ""
    nearby_text: str = ""

    path_obj = Path(path)
    file_name = path_obj.stem
    split_file_name = file_name.split("-")
    company = split_file_name[0]
    year = split_file_name[1][0:4]
    file_path = path

    # Get all the divs
    div_tags = soup.find_all("div")
    for div in div_tags:
        # print(div.prettify(), "\n")
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
                    if span_text:
                        nearby_text = span_text

                    IS_HEADING: bool = (
                        props.font_weight == "700" or props.font_size == "10"
                    ) and props.font_style == "italic"
                    IS_SECTION: bool = (
                        props.font_weight == "700" and span_text != "\u2022"
                    )
                    IS_PLAIN_TEXT: bool = props.font_weight == "400"
                    VALID_CITATION: bool = (
                        current_text != ""
                        and current_item != ""
                        and current_text != None
                    )

                    item = re.search("ITEM", span_text) or re.search("Item", span_text)
                    if item != None and (
                        props.font_weight == "700" or props.font_size == "14"
                    ):
                        if VALID_CITATION:
                            citations.append(
                                Citation(
                                    item=current_item,
                                    section=current_section,
                                    heading=current_heading,
                                    text=current_text,
                                    company=company,
                                    year=year,
                                    source=file_name,
                                    file_path=file_path,
                                )
                            )
                        current_item = span_text
                        # If this changes, all below values should change asw
                        current_section = ""
                        current_heading = ""
                        current_text = ""

                    elif IS_HEADING:
                        if VALID_CITATION:
                            citations.append(
                                Citation(
                                    item=current_item,
                                    section=current_section,
                                    heading=current_heading,
                                    text=current_text,
                                    company=company,
                                    year=year,
                                    source=file_name,
                                    file_path=file_path,
                                )
                            )
                        current_heading = span_text
                        current_text = ""

                    elif IS_SECTION:
                        if VALID_CITATION:
                            citations.append(
                                Citation(
                                    item=current_item,
                                    section=current_section,
                                    heading=current_heading,
                                    text=current_text,
                                    company=company,
                                    year=year,
                                    source=file_name,
                                    file_path=file_path,
                                )
                            )
                            current_section = span_text
                        elif current_text == "":
                            current_section = (
                                (current_section + " " + span_text).strip()
                                if current_section
                                else span_text
                            )
                        else:
                            current_section = span_text
                        current_heading = ""
                        current_text = ""

                    elif IS_PLAIN_TEXT:
                        current_text = current_text + span_text + "\n"

                except Exception as e:
                    print(div.prettify())
                    print("found error:", e)

            elif child_tag.name == "table":
                result = extract_table(child_tag)

                if result:
                    flattened_text, markdown_text = result
                    # print(table_as_markdown)
                    citations.append(
                        Citation(
                            item=current_item,
                            section=current_section,
                            heading=current_heading,
                            text=markdown_text,
                            nearby_text=nearby_text,
                            company=company,
                            year=year,
                            is_table=True,
                            flatten_table=flattened_text,
                            source=file_name,
                            file_path=file_path,
                        )
                    )

    # print(citations)
    return citations


if __name__ == "__main__":
    data_directory = "../data"

    path = "../data/amd-20251227.html"
    citations = get_citations(path)

    output_file_name = "citations.json"
    pretty = "\n\n".join(json.dumps(c.model_dump(), indent=2) for c in citations)
    pathlib.Path(output_file_name).write_text(pretty)
    print("Citations for", path, "can be viewed on,", output_file_name)
