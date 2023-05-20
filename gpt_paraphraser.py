import json
import openai
from time import sleep
from datetime import datetime


def prompt_api(prompt):
    while True:
        try:
            return openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}])
        except openai.error.RateLimitError:
            print("Rate Limited")
            sleep(20)
        except openai.error.APIError:
            print("API Error")
            sleep(10)


def paraphrase_sentence(sentence):
    prompt = f"Rephrase '{sentence}' keeping its meaning but changing some words."
    return prompt_api(prompt)["choices"][0]["message"]["content"]


def paraphrase_pair(data, original):
    if data["premise"] and data["hypothesis"]:
        return data
    else:
        return {"label": data["label"],
                "premise": paraphrase_sentence(original["premise"]),
                "hypothesis": paraphrase_sentence(original["hypothesis"])}


def paraphrase_corpus(inp_path, out_path):
    with open(inp_path, "r") as inp_f:
        with open(out_path, "w") as out_f:
            for id_, row in enumerate(inp_f):
                data = json.loads(row)
                result = {"label": data["label"],
                          "premise": paraphrase_sentence(data["premise"]),
                          "hypothesis": paraphrase_sentence(data["hypothesis"])}
                json.dump(result, out_f)
                out_f.write('\n')

                # checking progress
                if id_ % 100 == 0:
                    print(id_, datetime.now())


def main():
    # set API key
    openai.api_key = ""

    paraphrase_corpus("data/original/multinli_dev_matched.jsonl",
                      "data/gpt/multinli_dev_matched_gpt.jsonl")

    paraphrase_corpus("data/original/multinli_dev_mismatched.jsonl",
                      "data/gpt/multinli_dev_mismatched_gpt.jsonl")


if __name__ == '__main__':
    main()
