
import sys

from sentence_transformers import SentenceTransformer
from chunking import chunking_workflow
from eval import eval_workflow
from query import query_workflow

if __name__ == "__main__":
    workflow = sys.argv[1] if len(sys.argv) > 1 else "query"
    chunking_workflow()
    if workflow == "eval":
        eval_workflow()
    else:
        query_workflow()

