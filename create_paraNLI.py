import json


def add_sentences_to_dict(result, t):
    for paraphraser in ["bart", "gpt", "pegasus"]:
        filename = f"data/{paraphraser}/multinli_dev_{t}_{paraphraser}"
        with open(filename + ".jsonl", "r") as f:
            for id_, row in enumerate(f):
                data = json.loads(row)
                for sentence in ["premise", "hypothesis"]:
                    result[id_][paraphraser + "_text_" + sentence] = \
                        data[sentence]
    return result


def add_scores_to_dict(result, t):
    for paraphraser in ["bart", "pegasus", "gpt"]:
        for score in [["uniEval", "overall"],
                      ["bertscore", "F1"],
                      ["bleu_score", "bleu_score"]]:
            filename = f"data/{paraphraser}/multinli_dev_{t}_{paraphraser}_{score[0]}"
            with open(filename + ".jsonl", "r") as f:
                sentence_type = 0
                sentence = ["premise", "hypothesis"]
                i = 0
                for id_, row in enumerate(f):
                    data = json.loads(row)
                    i -= sentence_type
                    result[i][paraphraser + "_" + score[0] + "_" + sentence[sentence_type]] = \
                        data[score[1]]
                    sentence_type = 1 - sentence_type
                    i += 1
    return result


def get_all_scores_and_sentences(t):
    result = {}
    with open("data/original/multinli_dev_" + t + ".jsonl", "r") as f:
        for id_, row in enumerate(f):
            data = json.loads(row)
            result[id_] = data

    add_sentences_to_dict(result, t)
    add_scores_to_dict(result, t)
    return result


def calculate_score(value, sentence_type, paraphraser):
    if value[paraphraser + "_bleu_score_" + sentence_type] == 1:
        return 0
    return value[paraphraser + "_uniEval_" + sentence_type] * 0.5 + \
           value[paraphraser + "_bertscore_" + sentence_type] * 0.5


def get_highest_score(scores):
    max_scores = [key for key, value in scores.items() if value == max(scores.values())]
    if "gpt" in max_scores:
        return "gpt"
    else:
        return max_scores[0]


def choose_best_paraphraser(scores_and_sentences):
    para_nli = {}
    for key, value in scores_and_sentences.items():
        para_nli[key] = {"label": value["label"]}
        for sentence_type in ["premise", "hypothesis"]:
            scores = {"bart": calculate_score(value, sentence_type, "bart"),
                      "pegasus": calculate_score(value, sentence_type, "pegasus"),
                      "gpt": calculate_score(value, sentence_type, "gpt")}
            highest_score = get_highest_score(scores)
            para_nli[key]["best_paraphraser_" + sentence_type] = highest_score
            if highest_score == 0:
                print("SCORE IS 0")
            if highest_score == 1:
                print("SCORE IS 1")
            para_nli[key][sentence_type] = scores_and_sentences[key][highest_score + "_text_" + sentence_type]
    return para_nli


def main():
    for t in ["matched", "mismatched"]:
        scores_and_sentences = get_all_scores_and_sentences(t)
        with open("data/result/paraNLI_" + t + "_all_data.jsonl", "w") as out_f:
            for key, value in scores_and_sentences.items():
                json.dump(value, out_f)
                out_f.write("\n")
        with open("data/result/paraNLI_" + t + ".jsonl", "w") as f:
            best_paraphraser = choose_best_paraphraser(scores_and_sentences)
            for key, value in best_paraphraser.items():
                json.dump(value, f)
                f.write('\n')


if __name__ == '__main__':
    main()
