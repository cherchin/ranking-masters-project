# roleplayer.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from prompt import (
    ROLEPLAYER_SYSTEM,
    roleplayer_prompt,
)


class Roleplayer:
    def __init__(self, model):
        self.model = model
    def generate_user_answer(
        self,
        persona,
        task,
        question,
        conversation,
    ):

        prompt = f"""
    {ROLEPLAYER_SYSTEM}

    {roleplayer_prompt(
        persona=persona,
        task=task,
        question=question,
        conversation=conversation
    )}
    """

        return self.model.generate(
            prompt,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
        )