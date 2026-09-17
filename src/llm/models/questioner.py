from prompt import (
    QUESTIONER_SYSTEM,
    questioner_prompt,
    response_prompt,
)

class Questioner:

    def __init__(self, model):
        self.model = model

    def generate_question(
        self,
        task,
        conversation,
        turn_number,
    ):

        system_prompt = QUESTIONER_SYSTEM.format(
            name="the user",
        )

        user_prompt = questioner_prompt(
            task=task,
            conversation=conversation,
            turn_number=turn_number,
        )

        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ]

        question = self.model.generate(
            messages,
            max_new_tokens=80,
            temperature=0.7,
            do_sample=True,
        )

        return question.strip()
    
    def answer(
        self,
        task,
        conversation,
    ):

        prompt = response_prompt(
            task=task,
            conversation=conversation,
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. "
                    "Answer the user's original request using "
                    "the preferences revealed in the conversation."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

        return self.model.generate(
            messages,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
        ).strip()