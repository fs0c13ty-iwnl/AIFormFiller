import io
from pdfminer.converter import TextConverter
from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from pdfrw import PdfReader


def read_pdf(pdf_file):
    resource_manager = PDFResourceManager()
    fake_file_handle = io.StringIO()
    converter = TextConverter(resource_manager, fake_file_handle, laparams=LAParams())
    page_interpreter = PDFPageInterpreter(resource_manager, converter)

    with open(pdf_file, 'rb') as fh:
        for page in PDFPage.get_pages(fh,
                                      caching=True,
                                      check_extractable=True):
            page_interpreter.process_page(page)

        text = fake_file_handle.getvalue()

    # close open handles
    converter.close()
    fake_file_handle.close()

    return text


def read_pdf_fields(pdf_file):
    pdf = PdfReader(pdf_file)
    fields = pdf.Info
    return fields


def compare_pdfs(pdf_file1, pdf_file2):
    text1 = read_pdf(pdf_file1)
    text2 = read_pdf(pdf_file2)
    fields1 = read_pdf_fields(pdf_file1)
    fields2 = read_pdf_fields(pdf_file2)

    total_fields = sum(1 for v in fields1.values() if v != '')
    common_fields = sum(1 for k, v in fields1.items() if fields2.get(k) == v and v != '')

    accuracy = common_fields / total_fields if total_fields else 0

    return accuracy

pdf_file1 = 'example_form1.pdf'
pdf_file2 = 'Output_form_result.pdf'

print(compare_pdfs(pdf_file1, pdf_file2))
