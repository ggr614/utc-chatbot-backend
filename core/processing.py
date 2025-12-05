import html2text
from utils.tokenizer import Tokenizer



class TextProcessor:
    def __init__(self):
        self.html_converter = html2text.HTML2Text()
        self.html_converter.ignore_links = True
        self.html_converter.ignore_images = True
        self.html_converter.ignore_tables = True
        self.html_converter.body_width = 0  # Disable word wrapping

    def process_text(self, text: str) -> str:
        # Convert HTML to Markdown, then clean up
        markdown_text = self.html_converter.handle(text)
        # Remove extra newlines, leading/trailing whitespace, and multiple spaces
        cleaned_text = " ".join(markdown_text.split()).strip()
        return cleaned_text
    
    def get_token_count(self, text: str) -> int:
        tokenizer = Tokenizer()
        return tokenizer.num_tokens_from_string(text)
    

