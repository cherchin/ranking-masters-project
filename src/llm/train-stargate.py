import json
import os
import random

from tqdm import tqdm
from models.questioner import Questioner
from models.roleplayer import Roleplayer
from models.oracle import Oracle
from models.scorer import Scorer
from models.languagemodel import LanguageModel
from prompt import format_conversation

# CONFIG

MODEL_NAME = "HuggingFaceTB/SmolLM2-360M-Instruct"

NUM_TRAJECTORIES = 5
MAX_TURNS = 5

OUTPUT_FILE = "stargate-data/stargate_iteration_1.json"

SEED = 42

random.seed(SEED)

def load_tasks():

    with open(
        "stargate-data/tasks.json",
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)
    
def load_personas():

    with open(
        "stargate-data/personas.json",
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)
personas = load_personas()


def simulate_conversation(
    questioner,
    roleplayer,
    task,
    persona,
):

    conversation = []

    for turn in range(1, MAX_TURNS + 1):

        history = format_conversation(
            conversation
        )

        question = questioner.generate_question(
            task=task,
            conversation=history,
            turn_number=turn,
        )

        answer = roleplayer.generate_user_answer(
            persona=persona["description"],
            task=task,
            question=question,
            conversation=history,
        )

        conversation.append({
            "role": "assistant",
            "content": question,
        })

        conversation.append({
            "role": "user",
            "content": answer,
        })

    return conversation

def generate_gold_responses(
    oracle,
    tasks,
):

    gold = {}

    print("\nGenerating gold responses...")

    for task in tqdm(tasks):

        for persona in personas:

            response = oracle.generate_gold_response(
                task=task["task"],
                persona=persona["description"],
            )

            key = (
                task["id"],
                persona["id"],
            )

            gold[key] = response

    return gold

def generate_candidates(
    questioner,
    roleplayer,
    tasks,
):
    candidates = []

    print("\nGenerating candidate conversations...")

    for task in tqdm(tasks):

        for persona in personas:

            for trajectory_id in range(
                NUM_TRAJECTORIES
            ):

                conversation = simulate_conversation(
                    questioner=questioner,
                    roleplayer=roleplayer,
                    task=task["task"],
                    persona=persona,
                )

                candidates.append({
                    "task_id": task["id"],
                    "persona_id": persona["id"],
                    "trajectory_id": trajectory_id,
                    "task": task["task"],
                    "persona": persona["description"],
                    "conversation": conversation,
                })

    return candidates

def filter_conversations(
    scorer,
    candidates,
    gold_responses,
):
    grouped = {}

    print("\nScoring conversations...")

    for candidate in tqdm(candidates):

        key = (
            candidate["task_id"],
            candidate["persona_id"],
        )

        conversation_text = format_conversation(
            candidate["conversation"]
        )

        gold = gold_responses[key]

        score = scorer.score_conversation(
            task=candidate["task"],
            conversation=conversation_text,
            gold_response=gold,
        )

        candidate["score"] = score

        if key not in grouped:
            grouped[key] = []

        grouped[key].append(candidate)


    # Select highest-scoring trajectory

    selected = []

    for key, trajectories in grouped.items():

        best = max(
            trajectories,
            key=lambda x: x["score"]
        )

        selected.append(best)

    return selected

def add_response_regularization(
    questioner,
    selected,
):

    training_examples = []

    print("\nGenerating response-regularization examples...")

    for example in tqdm(selected):

        conversation_text = format_conversation(
            example["conversation"]
        )

        response = questioner.answer(
            task=example["task"],
            conversation=conversation_text,
        )

        example["response"] = response

        training_examples.append(
            example
        )

    return training_examples

def save_dataset(
    examples,
    filename,
):

    os.makedirs(
        os.path.dirname(filename),
        exist_ok=True,
    )

    with open(
        filename,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            examples,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"\nSaved {len(examples)} examples to {filename}"
    )

def main():

    print("\nInitializing models...")

    shared_model = LanguageModel(
        model_name=MODEL_NAME
    )

    questioner = Questioner(
        model=shared_model
    )

    roleplayer = Roleplayer(
        model=shared_model
    )

    oracle = Oracle(
        model=shared_model
    )

    scorer = Scorer(
        model=shared_model
    )

    tasks = load_tasks()

    print(
        f"\nTasks: {len(tasks)}"
    )

    print(
        f"Personas: {len(personas)}"
    )

    print(
        f"Trajectories per pair: "
        f"{NUM_TRAJECTORIES}"
    )

    gold_responses = generate_gold_responses(
        oracle=oracle,
        tasks=tasks,
    )

    candidates = generate_candidates(
        questioner=questioner,
        roleplayer=roleplayer,
        tasks=tasks,
    )

    print(
        f"\nGenerated "
        f"{len(candidates)} conversations."
    )

    selected = filter_conversations(
        scorer=scorer,
        candidates=candidates,
        gold_responses=gold_responses,
    )

    print(
        f"\nSelected "
        f"{len(selected)} best conversations."
    )

    selected = add_response_regularization(
        questioner=questioner,
        selected=selected,
    )

    save_dataset(
        examples=selected,
        filename=OUTPUT_FILE,
    )

    print("SELECTED EXAMPLES")

    for example in selected[:3]:

        print(
            f"\nTask: {example['task']}"
        )

        print(
            f"Score: {example['score']:.3f}"
        )

        print(
            "\nConversation:"
        )

        print(
            format_conversation(
                example["conversation"]
            )
        )

        print(
            "\nGenerated response:"
        )

        print(
            example["response"]
        )


if __name__ == "__main__":
    main()