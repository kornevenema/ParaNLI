import json
import random


def main():
    random.seed(0)
    result = {}
    data = {}
    for t in ["matched", "mismatched"]:
        with open("../data/result/paraNLI_" + t + ".jsonl", encoding="utf-8") as f:
            rows = [[enumerator, json.loads(line)["premise"], json.loads(line)["hypothesis"]]
                    for enumerator, line in enumerate(f.readlines())]
            data[t] = random.sample(rows, 250)
    with open("sentences.jsonl", "w", encoding="utf-8") as f_out:
        for t in data:
            for row in data[t]:
                row = {"enumerator": row[0],
                       "type": t,
                       "premise": row[1],
                       "hypothesis": row[2]}
                json.dump(row, f_out)
                f_out.write("\n")




if __name__ == '__main__':
    main()
