from bs4 import BeautifulSoup
from charset_normalizer import from_path
import re
import pandas as pd

content = str(from_path("../data/amd-20251227.html").best())
soup = BeautifulSoup(content, 'html.parser')

text_only = soup.get_text()
whitespace_removed = re.sub(r'\s+', ' ', text_only)

# get tables
tables = soup.find_all('table')
# tables_as_dataframe = pd.read_html("../data/nvda-20260125.html")
# tables_as_dataframe =  tables_as_dataframe
# markdown_tables = []
# for table in tables_as_dataframe:
#     table = table.dropna(axis=0, how="all").dropna(axis=1, how="all")
#     table = table.loc[:, ~table.T.duplicated()]
#     print(table)

tables = soup.find_all('table')
print(tables)

# print(tables_as_dataframe)

# print(whitespace_removed)