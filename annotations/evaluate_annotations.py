import json
import random
from sklearn import metrics


def print_random_example(labels):
    # pred_label == class. before para
    for i in range(0, 1000):
        r = random.randrange(0, len(labels[0]))
        result = []
        for item in labels:
            result.append(item[r])
        if result[1] != result[3] and len(result[2][0]) < 140:
            print("pred. bef. para. label: {0}".format(result[1]))
            print()
            print("human label: {0}".format(result[0]))
            print("pred. aft. para. label: {0}".format(result[3]))
            print("premise:     {0}".format(result[2][0]))
            print("hypothesis:  {0}".format(result[2][1]))
            print("-" * 50)
            return


def main():
    for t in ["matched", "mismatched"]:
        labels = [[], [], [], []]
        with open(f"paraNLI_{t}_annotations.jsonl", mode="r", encoding="utf-8") as f:
            with open(f"paraNLI_{t}_annotations_predictions.jsonl", mode="r", encoding="utf-8") as f_pred:
                for item1, item2 in zip(f, f_pred):
                    labels[0].append(json.loads(item1)["label"])
                    labels[1].append(json.loads(item2)["label"])
                    labels[3].append(json.loads(item1)["real_pred"])
                    labels[2].append([json.loads(item1)["premise"],
                                      json.loads(item1)["hypothesis"]])
        # print("{0:<20}{1:.3g}".format("accuracy", metrics.f1_score(labels[0], labels[1],
        #                                                            average="weighted")))
        print_random_example(labels)


if __name__ == '__main__':
    main()
