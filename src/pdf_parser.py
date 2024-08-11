import re
import nltk
import PyPDF2
from nltk.corpus import stopwords

nltk.download('stopwords')

class PDFExtractor:
    def __init__(self) -> None:
        pass

    def extract_text_from_pdf(self, pdf_path):
        with open(pdf_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = []
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text.append(page.extract_text())
            return text

class TextCleaner:
    def __init__(self) -> None:
        pass

    @staticmethod
    def clean_text(text):
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        return text

    @staticmethod
    def remove_stop_words(text):
        stop_words = set(stopwords.words('english'))
        words = text.split()
        filtered_words = [word for word in words if word.lower() not in stop_words]
        return ' '.join(filtered_words)

class TextChunker:
    def __init__(self) -> None:
        pass

    @staticmethod
    def split_into_chunks(text_list, sentences_per_chunk=10):
        chunks = []
        for text in text_list:
            # Split text based on periods and new paragraphs
            sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s|\n\n', text)
            for i in range(0, len(sentences), sentences_per_chunk):
                chunk = '. '.join(sentences[i: i + sentences_per_chunk])
                chunks.append(chunk)
        return chunks

class TextProcessor:
    def __init__(self) -> None:
        self.pdf_extractor = PDFExtractor()
        self.text_cleaner = TextCleaner()
        self.text_chunker = TextChunker()

    def process_pdf(self, pdf_path):

        text_list = self.pdf_extractor.extract_text_from_pdf(pdf_path)
        chunks = self.text_chunker.split_into_chunks(text_list)
        cleaned_chunks = [self.text_cleaner.remove_stop_words(self.text_cleaner.clean_text(chunk)) for chunk in chunks]
        return cleaned_chunks

