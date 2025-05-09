from abc import ABC, abstractmethod
import re
import string


class DataCleaner(ABC):
    def clean_text(self):
        self.lower()
        self.remove_matched_text()
        self.remove_urls()
        self.remove_html_tags()
        self.remove_punctuation()
        self.remove_newlines()
        self.remove_words_with_digits()
        self.normalize_spaces()
        return self

    @abstractmethod
    def lower(self):
        pass

    @abstractmethod
    def remove_matched_text(self):
        pass

    @abstractmethod
    def remove_urls(self):
        pass

    @abstractmethod
    def remove_html_tags(self):
        pass

    @abstractmethod
    def remove_punctuation(self):
        pass

    @abstractmethod
    def remove_newlines(self):
        pass

    @abstractmethod
    def remove_words_with_digits(self):
        pass

    @abstractmethod
    def normalize_spaces(self):
        pass


class TextCleaner(DataCleaner):
    def __init__(self, text: str):
        self.text = text

    def lower(self):
        self.text = self.text.lower()

    def remove_matched_text(self):
        self.text = re.sub(r'\[.*?\]', "", self.text)

    def remove_urls(self):
        self.text = re.sub(r'https?://\S+|www\.\S+', '', self.text)

    def remove_html_tags(self):
        self.text = re.sub(r'<.*?>', '', self.text)

    def remove_punctuation(self):
        self.text = re.sub(r'[%s]' % re.escape(
            string.punctuation), '', self.text)

    def remove_newlines(self):
        self.text = re.sub(r'\n', ' ', self.text)

    def remove_words_with_digits(self):
        self.text = re.sub(r'\w*\d\w*', '', self.text)

    def normalize_spaces(self):
        self.text = re.sub(r'\s+', ' ', self.text).strip()

    def get_cleaned_text(self):
        return self.text
