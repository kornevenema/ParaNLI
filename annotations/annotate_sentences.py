import json
from os.path import exists


def get_lines(path):
    with open(path, "r") as f:
        return [json.loads(row) for row in f]


def ask_input():
    while True:
        inp = input()
        if inp in ["e", "c", "n", "1", "2", "3"]:
            return inp
        else:
            print("input must be e, c, or n: [e -> entailment] [c -> contradiction] [n -> neutral]")


def inp_to_file(inp, f, line):
    mappings = {"e": "entailment",
                "1": "entailment",
                "n": "neutral",
                "2": "neutral",
                "c": "contradiction",
                "3": "contradiction"}
    outp = {"label": mappings[inp],
            "enumerator": line["enumerator"],
            "type": line["type"],
            "premise": line["premise"],
            "hypothesis": line["hypothesis"]}
    json.dump(outp, f)
    f.write("\n")
    

def main():
    lines = get_lines("sentences.jsonl")
    outf = "annotated_data.jsonl"
    if not exists(outf):
        with open(outf, "w") as f:
            print("Classify the following sentences: [e -> entailment] [c -> contradiction] [n -> neutral]")
            for line in lines:
                print("-" * 50)
                print("premise:    ", line["premise"])
                print("hypothesis: ", line["hypothesis"])
                inp_to_file(ask_input(), f, line)
    else:
        with open(outf, "a+") as f:
            print("Classify the following sentences: [e -> entailment] [c -> contradiction] [n -> neutral]")
            for count, line in enumerate(lines):
                if count < len(get_lines("annotated_data.jsonl")):
                    continue
                else:
                    print("-" * 50)
                    print("premise:    ", line["premise"])
                    print("hypothesis: ", line["hypothesis"])
                    inp_to_file(ask_input(), f, line)
                    




if __name__ == "__main__":
    main()
