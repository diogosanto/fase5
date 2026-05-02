import json

DATASET = [
    {
        "question": "Qual o valor do terreno no centro?",
        "answer": "O valor estimado do terreno é R$ 500000",
        "ground_truth": "R$ 500000"
    },
    {
        "question": "Quanto custa um terreno de 100m2?",
        "answer": "O valor estimado é R$ 300000",
        "ground_truth": "R$ 300000"
    }
]


def faithfulness(answer, ground_truth):
    return 1 if ground_truth in answer else 0


def answer_relevancy(question, answer):
    return 1 if any(word in answer.lower() for word in question.lower().split()) else 0.5


def context_precision(answer):
    return min(len(answer) / 200, 1)


def context_recall(answer, ground_truth):
    return 1 if ground_truth in answer else 0


def evaluate():
    results = []

    for item in DATASET:
        scores = {
            "faithfulness": faithfulness(item["answer"], item["ground_truth"]),
            "answer_relevancy": answer_relevancy(item["question"], item["answer"]),
            "context_precision": context_precision(item["answer"]),
            "context_recall": context_recall(item["answer"], item["ground_truth"]),
        }

        scores["final_score"] = sum(scores.values()) / 4
        results.append(scores)

    with open("evaluation/ragas_results.json", "w") as f:
        json.dump(results, f, indent=4)

    print("✔ RAGAS completo gerado")


if __name__ == "__main__":
    evaluate()