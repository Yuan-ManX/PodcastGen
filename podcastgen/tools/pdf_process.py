import logging
import os
import unicodedata
from dataclasses import dataclass
from typing import Optional
import pymupdf


@dataclass
class PDFExtractorConfig:
    remove_accents: bool = True


class PDFExtractor:
    def __init__(self, config: Optional[PDFExtractorConfig] = None):
        """
        Initializes the PDFExtractor with optional configuration.

        Args:
            config (PDFExtractorConfig, optional): Configuration for the PDF extractor.
                                                   Defaults to None.
        """
        if config is None:
            config = PDFExtractorConfig()
        self.config = config

    def extract_pdf_content(self, file_path: str) -> str:
        """
        Extracts text content from a PDF file, handling foreign and special characters.
        Accents are removed based on the configuration.

        Args:
            file_path (str): Path to the PDF file.

        Returns:
            str: Extracted text content with accents removed if configured.

        Raises:
            FileNotFoundError: If the PDF file does not exist.
            ValueError: If the file_path is not a valid PDF.
            Exception: For any other exceptions during extraction.
        """
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")

        if fitz is None:
            raise ImportError("PyMuPDF (fitz) is not installed. Cannot extract PDF content.")

        try:
            with fitz.open(file_path) as doc:
                pages = [page.get_text() for page in doc]
                content = " ".join(pages)

            if self.config.remove_accents:
                # Normalize the text to handle special characters and remove accents
                normalized_content = unicodedata.normalize('NFKD', content)
                # Filter out combining characters to remove accents
                cleaned_content = ''.join(
                    char for char in normalized_content
                    if not unicodedata.combining(char)
                )
                return cleaned_content
            else:
                return content
        except Exception as e:
            logging.error(f"Error extracting PDF content from {file_path}: {e}")
            raise

def run_pdf_extractor(seed: int = 42) -> None:
    """
    Tests the PDFExtractor with a specific PDF file and prints the extracted content.

    Args:
        seed (int, optional): Random seed for reproducibility. Defaults to 42.
    """
    # Set the random seed
    import random
    random.seed(seed)

    # Get the absolute path of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the path to the PDF file
    pdf_path = os.path.abspath(os.path.join(script_dir, '..', '..', 'tests', 'data', 'file.pdf'))

    if not os.path.isfile(pdf_path):
        logging.error(f"The PDF file {pdf_path} does not exist.")
        return

    extractor_config = PDFExtractorConfig(remove_accents=True)
    extractor = PDFExtractor(config=extractor_config)

    try:
        content = extractor.extract_pdf_content(pdf_path)
        print("PDF content extracted successfully:")
        print(content[:500] + "..." if len(content) > 500 else content)
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    run_pdf_extractor()
