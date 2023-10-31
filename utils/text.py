from natasha import (
    Segmenter,
    MorphVocab,
    NewsEmbedding,
    NewsMorphTagger,
    NewsSyntaxParser,
    Doc
)


def lemmatize_text(text: str) -> Doc:
    if not lemmatize_text.initialized:
        lemmatize_text.initialized = True
        lemmatize_text.segmenter = Segmenter()
        lemmatize_text.morph_vocab = MorphVocab()
        lemmatize_text.emb = NewsEmbedding()

        lemmatize_text.emb = NewsEmbedding()
        lemmatize_text.morph_tagger = NewsMorphTagger(lemmatize_text.emb)
        lemmatize_text.syntax_parser = NewsSyntaxParser(lemmatize_text.emb)

    doc = Doc(text)
    doc.segment(lemmatize_text.segmenter)
    doc.tag_morph(lemmatize_text.morph_tagger)
    for token in doc.tokens:
        token.lemmatize(lemmatize_text.morph_vocab)
    return doc


lemmatize_text.initialized = False


def join_lemmatized_doc(doc: Doc) -> str:
    return " ".join(token.lemma for token in doc.tokens)
