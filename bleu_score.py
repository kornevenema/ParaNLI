from nltk.translate import bleu_score
from nltk.tokenize import word_tokenize
import json


def source_sentences_to_list(path):
    out = []
    with open(path) as f:
        for row in f:
            data = json.loads(row)
            out.append(data["premise"])
            out.append(data["hypothesis"])
    return out


def scores_to_file(path, bleu_scores):
    with open(path, "w") as f:
        for i in range(0, len(bleu_scores)):
            json.dump({"bleu_score": bleu_scores[i]}, f)
            f.write('\n')


def get_bleu(references, hypothesises):
    smoothing = bleu_score.SmoothingFunction()
    return [bleu_score.sentence_bleu([word_tokenize(x)], word_tokenize(y), smoothing_function=smoothing.method3) for x, y in zip(references, hypothesises)]


def calculate_bleu_corpus(ref_path, hypo_path, out_path):
    bleu_scores = get_bleu(source_sentences_to_list(ref_path), source_sentences_to_list(hypo_path))
    scores_to_file(out_path, bleu_scores)


def main():
    calculate_bleu_corpus("data/original/multinli_dev_matched.jsonl",
                          "data/bart/multinli_dev_matched_bart.jsonl",
                          "data/bart/multinli_dev_matched_bart_bleu_score.jsonl")

    calculate_bleu_corpus("data/original/multinli_dev_mismatched.jsonl",
                          "data/bart/multinli_dev_mismatched_bart.jsonl",
                          "data/bart/multinli_dev_mismatched_bart_bleu_score.jsonl")

    calculate_bleu_corpus("data/original/multinli_dev_matched.jsonl",
                          "data/pegasus/multinli_dev_matched_pegasus.jsonl",
                          "data/pegasus/multinli_dev_matched_pegasus_bleu_score.jsonl")

    calculate_bleu_corpus("data/original/multinli_dev_mismatched.jsonl",
                          "data/pegasus/multinli_dev_mismatched_pegasus.jsonl",
                          "data/pegasus/multinli_dev_mismatched_pegasus_bleu_score.jsonl")

    calculate_bleu_corpus("data/original/multinli_dev_matched.jsonl",
                          "data/gpt/multinli_dev_matched_gpt.jsonl",
                          "data/gpt/multinli_dev_matched_gpt_bleu_score.jsonl")

    calculate_bleu_corpus("data/original/multinli_dev_mismatched.jsonl",
                          "data/gpt/multinli_dev_mismatched_gpt.jsonl",
                          "data/gpt/multinli_dev_mismatched_gpt_bleu_score.jsonl")


if __name__ == '__main__':
    main()
