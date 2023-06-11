import json
import pandas as pd
from sklearn.metrics import cohen_kappa_score


def get_annotations(t):
    result = []
    with open(f"../data/result/multinli_dev_{t}_predictions.jsonl", mode="r", encoding="utf-8") as f4:
        with open(f"../data/original/multinli_dev_{t}.jsonl", mode="r", encoding="utf-8") as f3:
            with open(f"../data/result/paraNLI_{t}_predictions.jsonl", mode="r", encoding="utf-8") as f5:
                original = [json.loads(row) for row in f3]
                original2 = [json.loads(row) for row in f4]
                original3 = [json.loads(row) for row in f5]
                with open("annotated_data.jsonl", mode="r", encoding="utf-8") as f1:
                    with open("annotated_data2.jsonl", mode="r", encoding="utf-8") as f2:
                        ann1 = [json.loads(row) for row in f1]
                        ann2 = [json.loads(row) for row in f2]
                        for item1, item2 in zip(ann1, ann2):
                            gold_label = [row["label"] for count, row in enumerate(original)
                                          if item1["enumerator"] == count]
                            pred_labels = [row["label"] for count, row in enumerate(original2)
                                           if item1["enumerator"] == count]
                            pred_labels2 = [row["label"] for count, row in enumerate(original3)
                                            if item1["enumerator"] == count]
                            result.append(
                                [item1["label"], item2["label"], gold_label[0], item1["type"], item1["premise"],
                                 item1["hypothesis"], item1["enumerator"], pred_labels[0], pred_labels2[0]])
            return pd.DataFrame(result, columns=["ann1", "ann2", "gold", "type", "premise", "hypothesis", "enumerator",
                                                 "pred_label", "real_pred"])


def choose_label(labels):
    if labels[0] == labels[1]:
        return labels[0]
    elif labels[1] == labels[2]:
        return labels[1]
    elif labels[0] == labels[2]:
        return labels[0]
    else:
        return labels[2]


def main():
    for t in ["matched", "mismatched"]:
        labels = get_annotations(t)
        print(cohen_kappa_score(labels.loc[labels['type'] == t]["ann1"],
                                labels.loc[labels['type'] == t]["gold"]))
        print(cohen_kappa_score(labels.loc[labels['type'] == t]["ann2"],
                                labels.loc[labels['type'] == t]["gold"]))
        print(cohen_kappa_score(labels.loc[labels['type'] == t]["ann1"],
                                labels.loc[labels['type'] == t]["ann2"]))

        with open(f"paraNLI_{t}_annotations.jsonl", mode="w", encoding="utf-8") as f:
            for index, item in labels.loc[labels['type'] == t].iterrows():
                out = {"enumerator": item["enumerator"],
                       "label": choose_label([item["ann1"], item["ann2"], item["gold"]]),
                       "pred_label": item["pred_label"],
                       "orig_label": item["gold"],
                       "premise": item["premise"],
                       "real_pred": item["real_pred"],
                       "hypothesis": item["hypothesis"]}
                json.dump(out, f)
                f.write("\n")


if __name__ == '__main__':
    main()
