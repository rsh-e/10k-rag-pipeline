
EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1.5"
COLLECTION_NAME = "data_store"
ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2" 
STORAGE_PATH = "../storage"
DOCUMENT_DB_NAME = "../storage/document_ledger.db" 
NUMBER_OF_CITATIONS = 50
TOP_K = 5

COMPANIES = {
    "amd": ["amd", "advanced micro devices"],
    "axp": ["axp", "american express", "amex"],
    "cop": ["cop", "conocophilips", "conoco philips", "conoco"],
    "cvx": ["cvx", "chevron"],
    "goog": ["goog", "google", "alphabet"],
    "jnj": ["jnj", "johnson", "j&j"],
    "ko": ["ko", "coca cola", "coca-cola", "coke"],
    "meta": ["meta", "facebook"],
    "nvda": ["nvda", "nvidia"],
    "pep": ["pep", "pepsico", "pepsi"],
}

NLTK_MODULES = {
    "tokenizers/punkt": "punkt", 
    "tokenizers/punkt_tab": "punkt_tab",
    "corpora/stopwords": "stopwords"
}

FINANCIAL_STOP_WORDS= [
    "consolidated",
    "statement",
    "report",
    "table",
    "million",
    "fiscal",
    "results",
]

TABLE_WORDS = ["table", "statement", "statements", "tables"]
MODEL = "groq/compound" 
ROLE = "user"
TEMPERATURE = 1

MAX_COMPLETION_TOKENS = 1000
TOP_P = 1