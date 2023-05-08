import json


def extract_sentences_and_label(path, out_path):
    with open(out_path, "w", encoding="utf-8") as out_f:
        with open(path, encoding="utf-8") as f:
            for id_, row in enumerate(f):
                data = json.loads(row)
                new_data = {"label": data["gold_label"],
                            "premise": data["sentence1"],
                            "hypothesis": data["sentence2"],
                            }
                json.dump(new_data, out_f)
                out_f.write('\n')


def main():
    extract_sentences_and_label(
        "data/original/multinli_1.0_dev_matched.jsonl",
        "data/original/multinli_dev_matched.jsonl")
    extract_sentences_and_label(
        "data/original/multinli_1.0_dev_mismatched.jsonl",
        "data/original/multinli_dev_mismatched.jsonl")


if __name__ == '__main__':
    main()
