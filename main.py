import os
from openai import OpenAI, APIError, RateLimitError
import time
import numpy as np
import pdfrw
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdftypes import PDFObjRef
from config import OPENAI_API_KEY, EMBEDDING_FILE, PDF_FILES, INPUT_PDF_PATH, OUTPUT_PDF_PATH, OPENAI_MODEL, MAX_RETRIES, RETRY_DELAY, TOP_N, SIMILARITY_THRESHOLD

# Check if OPENAI_API_KEY has been correctly set ('none' means failed)
# Adding Your API Key to the environment variables using export OPENAI_API_KEY="YOUR OPENAI API KEY"
print(os.getenv("OPENAI_API_KEY"))

embedding_file = EMBEDDING_FILE
embedding_dict = {}

start_program = time.time()
start_loading_embeddings = time.time()
with open(embedding_file, 'r', encoding='utf-8') as file:
    for line in file:
        values = line.split()
        word = values[0]
        vector = np.array(values[1:], dtype='float32')
        embedding_dict[word] = vector

print(f"Total Words in Dict: {len(embedding_dict)}")
end_loading_embeddings = time.time()


def extract_form_fields(PDF_FILES):
    """
    Extract FIELD VALUES from PDFs.

    Args:
        pdf_file (str): Path to the PDF file.

    Returns:
        dict: A dictionary containing the extracted form fields and their values.
    """

    def deal_null(t):
        """
        Handle null values in the PDF form fields.

        Args:
            t: The value to be processed.

        Returns:
            str: The processed value.
        """
        try:
            if t is not None:
                t = t.decode('utf-8')
            else:
                t = ''
        except:
            if str(t)[0] == '/':
                t = str(t)[2:-1]
            else:
                t = ''
        return t

    with open(PDF_FILES, 'rb') as f:
        parser = PDFParser(f)
        document = PDFDocument(parser)
        catalog = document.catalog
        if 'AcroForm' not in catalog:
            return []
        acroform = catalog['AcroForm'].resolve() if isinstance(catalog['AcroForm'], PDFObjRef) else catalog['AcroForm']
        if not acroform:
            return []
        fields = acroform.get('Fields').resolve() if isinstance(acroform.get('Fields'), PDFObjRef) else acroform.get(
            'Fields')
        if not fields:
            return []
        result = {}
        for field in fields:
            key = field.resolve().get('T')
            key = deal_null(key)
            if not key: continue
            value = field.resolve().get('V')
            value = deal_null(value)
            if key in result:
                print(f'twice key')
            result[key] = value
        return result


def compute_similarity(vector, matrix, topN, threshold=0.5):
    """
    Compute the similarity between a vector and a matrix of vectors.

    Args:
        vector (numpy.ndarray): The input vector.
        matrix (numpy.ndarray): The matrix of vectors.
        topN (int): The number of top similar vectors to return.
        threshold (float, optional): The similarity threshold. Defaults to 0.5.

    Returns:
        numpy.ndarray: The indices of the top similar vectors.
    """
    similarity_scores = np.dot(matrix, vector) / (np.linalg.norm(matrix, axis=1) * np.linalg.norm(vector))
    top_indices = np.where(similarity_scores >= threshold)[0]
    top_indices = top_indices[np.argsort(similarity_scores[top_indices])[::-1]]
    top_indices = top_indices[:topN]
    if len(top_indices) and similarity_scores[top_indices[0]] == 1:
        return top_indices[:1]
    return top_indices


def match_topn_key_values(key, topN=TOP_N, threshold=SIMILARITY_THRESHOLD):
    """
    Match the top-N key-value pairs based on similarity.

    Args:
        key (str): The input key.
        topN (int, optional): The number of top key-value pairs to return. Defaults to 5.
        threshold (float, optional): The similarity threshold. Defaults to 0.5.

    Returns:
        dict: A dictionary containing the top-N key-value pairs.
    """
    output_dict = {}
    input_embed = [embedding_dict[w] for w in key.lower().split() if w in embedding_dict]
    if not input_embed:
        return None
    input_embed = np.array(input_embed).mean(0)
    topN_indices = compute_similarity(input_embed, pdf_key_vecs, topN=topN, threshold=threshold)
    topN_key_values = {pdf_keys[i]: results[pdf_keys[i]] for i in topN_indices}
    return topN_key_values


def cat_prompt(input_key, target_keys):
    """
    Concatenate the input key and target keys into a prompt string.

    Args:
        input_key (str): The input key.
        target_keys (dict): A dictionary containing the target keys and their values.

    Returns:
        str: The concatenated prompt string.
    """
    print("\n=======================================================")
    r_str = ['\nInput Fields Below']
    for k, v in target_keys.items():
        r_str.append(f'{k} is {v}')
    r_str.append('Output Field Below')
    r_str.append(f'{input_key}:')
    r_str.append(
        f'Deduce and fill in the output field {input_key} based on the input fields. Return N/A if inference not possible.No explaination needed Provide only the answer to be filled.')

    # r_str.append(f'1.Please fill in the Output Field[{input_key}] based on [Input Fields]:\n2.If input fields not helpful, return N/A \n3.Only need to answer the exact result\n4.No fail or succes sreasons needed \n5.Give answer in one line\n')
    r_str = '\n'.join(r_str)
    return r_str


def get_openai_result(prompt):
    """
    Get the result from the OpenAI API.

    Args:
        prompt (str): The input prompt.

    Returns:
        str: The result obtained from the OpenAI API.
    """
    # Initialize the OpenAI client using an environment variable for the API key
    client = OpenAI(api_key=os.getenv(OPENAI_API_KEY))

    retries = 0
    while retries < MAX_RETRIES:
        try:
            # Create a chat completion using the OpenAI client
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant with filling forms."},
                    {"role": "user", "content": prompt}
                ]
            )
            # Return the content of the first choice's message, stripped of any extra whitespace
            return response.choices[0].message.content.strip()

        except APIError as e:
            print(f"API error occurred: {e}")
            retries += 1
            time.sleep(RETRY_DELAY * retries)  # Incremental backoff
        except RateLimitError:
            retries += 1
            print(f"Rate limit exceeded, retrying after some delay ({retries}/5)")
            time.sleep(RETRY_DELAY * retries)  # Incremental backoff
    return "Unable to get response from OpenAI due to rate limits or API errors."


def write_data_to_pdf(input_pdf_path, output_pdf_path):
    """
    Write data to the output PDF file.

    Args:
        input_pdf_path (str): Path to the input PDF file.
        output_pdf_path (str): Path to the output PDF file.
    """
    template_pdf = pdfrw.PdfReader(input_pdf_path)
    for page in template_pdf.pages:
        annotations = page['/Annots']
        if annotations is None:
            continue
        for i, annotation in enumerate(annotations):
            if '/T' in annotation:  # Check if '/T' field exists in annotation
                input_key = annotation['/T'][1:-1].lower()
                target_keys_values = match_topn_key_values(input_key)
                prompt = ''
                if not target_keys_values:
                    value = ''
                    target_keys_values = ''
                elif len(target_keys_values) == 1:
                    value = list(target_keys_values.values())[0]
                    chatgpt_response = value
                else:
                    prompt = cat_prompt(input_key, target_keys_values)
                    chatgpt_response = get_openai_result(prompt)
                    value = chatgpt_response.split(':')[-1].strip()
                if not value:
                    value = ''
                print(f'\ninput_key: {input_key}\n')
                print(f'target_keys_values: {target_keys_values}\n')
                print(f'chatgpt prompt: {prompt}\n')
                print(f'chatgpt response: {chatgpt_response}\n')
                print("\n=======================================================")
                annotation.update(
                    pdfrw.PdfDict(V=value)
                )
    pdfrw.PdfWriter().write(output_pdf_path, template_pdf)

    print(f'Write file: ’{output_pdf_path}‘ finished!!!')


PDF_FILES = ["forms/example_form1.pdf", "forms/example_form2.pdf", "forms/example_form3.pdf"]
results = {}
for pdf_file in PDF_FILES:
    print(f'pdf_file: {pdf_file}')
    result = extract_form_fields(pdf_file)
    for k, v in result.items():
        k = k.lower()
        if k not in results:
            results[k] = v.strip()
        else:
            if v:
                results[k] = v.strip()

# Target pdf key vector
pdf_keys, pdf_key_vecs = [], []
for key, value in results.items():
    word_embed = [embedding_dict[w] for w in key.split() if w in embedding_dict]
    if not word_embed: continue
    pdf_keys.append(key)
    word_embed = np.array(word_embed).mean(0)
    pdf_key_vecs.append(word_embed)
    assert len(pdf_key_vecs) == len(pdf_keys)

pdf_key_vecs = np.array(pdf_key_vecs)

input_pdf_path = INPUT_PDF_PATH
output_pdf_path = OUTPUT_PDF_PATH

write_start = time.time()
write_data_to_pdf(input_pdf_path, output_pdf_path)
write_end = time.time()

end_program = time.time()
print(f"\nTotal Time to load embeddings: {end_loading_embeddings - start_loading_embeddings}s")
print(f"\nTotal Write to PDF time: {write_end - write_start}s")
print(f"\nTotal Runtime: {end_program - start_program}s")