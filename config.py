# config.py

# OpenAI API Key
OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"

# Embedding file path
EMBEDDING_FILE = "glove.6B.300d.txt"

# Input PDF files
PDF_FILES = [
    "forms/example_form1.pdf",
    "forms/example_form2.pdf",
    "forms/example_form3.pdf"
]

# Input and output PDF paths
INPUT_PDF_PATH = "forms/example_form1.pdf"
OUTPUT_PDF_PATH = "forms/Output_form_result.pdf"

# OpenAI model configuration
OPENAI_MODEL = "gpt-3.5-turbo"

# Retry configuration
MAX_RETRIES = 5
RETRY_DELAY = 10

# Similarity matching configuration
TOP_N = 5
SIMILARITY_THRESHOLD = 0.5