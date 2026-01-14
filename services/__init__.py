"""
Services paketi - Blind Assist için modüler servisler

Modüller:
- ocr_reader: MOD 2 - Metin Okuma (PaddleOCR)
- object_describer: MOD 3 - Nesne Tanımlama
- object_searcher: MOD 4 - Nesne Arama
- speech_service: TTS ses servisi
"""

# Lazy import - modüller ihtiyaç halinde yüklenir
__all__ = [
    'ocr_reader',
    'read_text',
    'describe_objects', 
    'get_turkish_name',
    'search_object',
    'get_available_objects',
]
