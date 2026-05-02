import json

def evaluate(question, answer):
    criterios = {
        "corretude": len(answer) > 15,
        "utilidade": "R$" in answer,
        "seguranca": "erro" not in answer.lower()
    }

    score = sum(criterios.values()) / 3

    return {
        "score": score,
        "criterios": criterios
    }


def run():
    dataset = [
        {
            "question": "valor terreno",
            "answer": "O valor estimado é R$ 450000"
        }
    ]

    results = []

    for item in dataset:
        results.append(evaluate(item["question"], item["answer"]))

    with open("evaluation/llm_judge_results.json", "w") as f:
        json.dump(results, f, indent=4)

    print("✔ LLM Judge finalizado")


if __name__ == "__main__":
    run()