from transformers import pipeline, MarianMTModel, MarianTokenizer
import spacy

sentiment_pipe = pipeline("sentiment-analysis")
summ_pipe = pipeline("summarization", model="facebook/bart-large-cnn")

nlp = spacy.load("en_core_web_sm")


def sentiment_analysis(text):
    return sentiment_pipe(text[:512])[0]


def summarize_text(text):
    return summ_pipe(text[:2000])[0]["summary_text"]


def translate_text(text, target):
    model_name = f"Helsinki-NLP/opus-mt-en-{target}"
    tok = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    batch = tok([text], return_tensors="pt", truncation=True)
    gen = model.generate(**batch)
    out = tok.batch_decode(gen, skip_special_tokens=True)

    return out[0]


def extract_entities(text):
    doc = nlp(text)
    return [
        {"text": ent.text, "label": ent.label_}
        for ent in doc.ents
    ]