import json
from collections import Counter
from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt
from nltk import word_tokenize


def get_labels(path_original, path_predictions):
    with open(path_original, mode="r", encoding="utf-8") as f_ori:
        with open(path_predictions, mode="r", encoding="utf-8") as f_pred:
            return pd.DataFrame([[json.loads(row1)["label"], json.loads(row2)["label"]]
                                 for row1, row2 in zip(f_ori, f_pred)
                                 if json.loads(row1)["label"] != "-"], columns=["true", "pred"])


def display_plot(labels):
    confusion_matrix = metrics.confusion_matrix(labels["true"], labels["pred"])
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix,
                                                display_labels=["contradiction", "neutral", "entailment"])
    cm_display.plot()
    plt.show()


def calculate_metrics(labels):
    print("{0:<20}{1:.3g}".format("accuracy", metrics.accuracy_score(labels["true"], labels["pred"])))
    print("{0:<20}{1:.3g}".format("precision", metrics.precision_score(labels["true"], labels["pred"],
                                                                       average="weighted", zero_division=1)))
    print("{0:<20}{1:.3g}".format("recall", metrics.recall_score(labels["true"], labels["pred"],
                                                                 average="weighted")))
    print("{0:<20}{1:.3g}".format("F1", metrics.f1_score(labels["true"], labels["pred"],
                                                         average="weighted")))
    #
    # print("{0:.3g} & {1:.3g} & {2:.3g}".format(
    #     # metrics.accuracy_score(labels["true"], labels["pred"]),
    #     metrics.precision_score(labels["true"], labels["pred"], average="weighted", zero_division=1),
    #     metrics.recall_score(labels["true"], labels["pred"], average="weighted"),
    #     metrics.f1_score(labels["true"], labels["pred"], average="weighted")
    # ))


def get_sentences(path):
    with open(path, mode="r", encoding="utf-8") as f:
        return [json.loads(row) for row in f]


def calc_stuff(sentences):
    tp = 0
    th = 0
    lp = 0
    lh = 0
    tokens_p = []
    tokens_h = []
    for sentence in sentences:
        wlp = word_tokenize(sentence["premise"])
        wlh = word_tokenize(sentence["hypothesis"])
        if len(wlp) > lp:
            lp = len(wlp)
        if len(wlh) > lh:
            lh = len(wlh)
        for word in wlp:
            if word not in tokens_p:
                tokens_p.append(word)
        for word in wlh:
            if word not in tokens_h:
                tokens_h.append(word)
        tp += len(wlp)
        th += len(wlh)
    print("mean sentence length: ", tp / len(sentences), th / len(sentences))
    print("Different Tokens: ", len(tokens_p), len(tokens_h))
    print("Length of stuff: ", len(sentences), len(sentences))
    print("longest: ", lp, lh)


def check_labels(labels):
    print(Counter(labels["true"]))
    print(Counter(labels["pred"]))


def run_evaluations(labels):
    check_labels(labels)
    calculate_metrics(labels)
    print()


def check_best_paraphraser(path):
    counts = {"bart_premise": 0,
              "bart_hypothesis": 0,
              "pegasus_premise": 0,
              "pegasus_hypothesis": 0,
              "gpt_premise": 0,
              "gpt_hypothesis": 0}
    with open(path, mode="r", encoding="utf-8") as f:
        for row in f:
            row = json.loads(row)
            for item in ["premise", "hypothesis"]:
                counts[row["best_paraphraser_" + item] + "_" + item] += 1
    print(counts)


def get_word_list(path):
    words = set()
    with open(path, mode="r", encoding="utf-8") as f:
        for sentence in [json.loads(row) for row in f]:
            for word in word_tokenize(sentence["premise"]):
                words.add(word)
            for word in word_tokenize(sentence["hypothesis"]):
                words.add(word)
    return words


def calc_words_used(path, original_word_list):
    total = 0
    words_used = 0
    new_words = 0
    for word in get_word_list(path):
        total += 1
        if word in original_word_list:
            words_used += 1
        else:
            new_words += 1

    print(f"total words: ", total)
    print(f"words used: ", words_used)
    print(f"new words: ", new_words)


def main():
    for t in ["matched", "mismatched"]:
        word_list = get_word_list(f"data/original/multinli_dev_{t}.jsonl")
        check_best_paraphraser(f'data/result/paraNLI_{t}.jsonl')

        # original dataset
        print("-" * 70)
        print("Type is " + t)
        print("Paraphraser is not used")
        labels = get_labels(f"data/original/multinli_dev_{t}.jsonl",
                            f'data/result/multinli_dev_{t}_predictions.jsonl')
        run_evaluations(labels)
        # calc_stuff(get_sentences(f"data/original/multinli_dev_{t}.jsonl"))
        # calc_words_used(f"data/original/multinli_dev_{t}.jsonl", word_list)

        # ParaNLI
        print("-" * 70)
        print("Type is " + t)
        print("Paraphraser is ParaNLI")
        labels = get_labels(f"data/original/multinli_dev_{t}.jsonl",
                            f'data/result/paraNLI_{t}_predictions.jsonl')
        run_evaluations(labels)
        display_plot(labels)

        # calc_stuff(get_sentences(f'data/result/paraNLI_{t}.jsonl'))
        # calc_words_used(f'data/result/paraNLI_{t}.jsonl', word_list)

        # solo paraphraser
        for paraphraser in ["bart", "pegasus", "gpt"]:
            labels = get_labels(f"data/original/multinli_dev_{t}.jsonl",
                                f'data/result/paraNLI_{t}_{paraphraser}_predictions.jsonl')
            print("-" * 70)
            print("Type is " + t)
            print("Paraphraser is " + paraphraser)
            run_evaluations(labels)
            # calc_stuff(get_sentences(f'data/result/paraNLI_{t}_{paraphraser}.jsonl'))
            # calc_words_used(f'data/result/paraNLI_{t}_{paraphraser}.jsonl', word_list)


if __name__ == '__main__':
    main()
