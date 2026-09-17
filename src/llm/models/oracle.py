# oracle.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from prompt import (
    ORACLE_SYSTEM,
    oracle_prompt,
)


class Oracle:


    def __init__(self, model):
        self.model = model

    def generate_gold_response(
        self,
        task,
        persona,
    ):

        prompt = f"""
    {ORACLE_SYSTEM}

    {oracle_prompt(
        task=task,
        persona=persona
    )}
    """

        return self.model.generate(
            prompt,
            max_new_tokens=300,
            temperature=0.0,
            do_sample=False,
        )