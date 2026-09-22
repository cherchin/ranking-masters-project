from prompt import ORACLE_SYSTEM, oracle_prompt


class Oracle:

    def __init__(self, model):
        self.model = model

    def generate_gold_response(
        self,
        task,
        persona,
    ):

        messages = [
            {
                "role": "system",
                "content": ORACLE_SYSTEM.strip(),
            },
            {
                "role": "user",
                "content": oracle_prompt(
                    task=task,
                    persona=persona,
                ).strip(),
            },
        ]

        return self.model.generate(
            messages,
            max_new_tokens=300,
            temperature=0.0,
            do_sample=False,
        ).strip()