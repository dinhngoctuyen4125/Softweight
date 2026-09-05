import json


def load_dforget(data_path, tokenizer, max_seq_length=512):
    """Load D_forget and tokenize it for feature extraction.

    The text fed to CodeBERT for each record is "probing input" + "y_neg".
    Returns the tokenized dataset and, per record, the list of deprecated APIs
    it belongs to (used to group the one-class SVMs).
    """
    # utf-8 is required: the code snippets contain non-cp1252 characters.
    with open(data_path, encoding="utf-8") as fp:
        records = json.load(fp)

    def preprocess_function(example):
        text = example["probing input"] + example["y_neg"]
        return tokenizer(text, padding="max_length", max_length=max_seq_length, truncation=True)

    dataset = [preprocess_function(r) for r in records]
    groups = [r["deprecated api"] for r in records]
    return dataset, groups
