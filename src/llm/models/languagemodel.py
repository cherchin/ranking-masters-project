import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class LanguageModel:

    def __init__(self, model_name):

        # Prefer GPU when available, otherwise fall back to CPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32

        print(f"Loading model on {self.device}")

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Using dtype: {dtype}")

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
            low_cpu_mem_usage=True,
        )

        self.model.to(self.device)
        self.model.eval()

        print("Model loaded successfully.")

    # Generation
    def generate(
        self,
        prompt,
        max_new_tokens=128, 
        temperature=0.7,
        do_sample=True,
    ):

        if isinstance(prompt, list):

            # Chat messages
            inputs = self.tokenizer.apply_chat_template(
                prompt,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )

        else:

            # Plain text prompt
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
            )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        print(
            f"Input shape: {inputs['input_ids'].shape}"
        )

        with torch.no_grad():
            if do_sample:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            else:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

        print(
            f"Output shape: {outputs.shape}"
        )

        print(
            f"Generated tokens: "
            f"{outputs.shape[-1] - inputs['input_ids'].shape[-1]}"
        )

        input_length = inputs["input_ids"].shape[-1]

        generated_tokens = outputs[
            0,
            input_length:
        ]

        print(
            f"Generated token IDs: "
            f"{generated_tokens.tolist()}"
        )

        response = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=False,
        )

        print(
            f"Raw decoded response: {repr(response)}"
        )

        return response.strip()

# scoring function: log probability of target (gold response) given prompt (conversation)
    def score(
            self,
            prompt,
            target,
        ):

            prefix = self.tokenizer(
                prompt,
                return_tensors="pt",
                add_special_tokens=False,
            )

            full = self.tokenizer(
                prompt + target,
                return_tensors="pt",
                add_special_tokens=False,
            )

            input_ids = full["input_ids"].to(
                self.device
            )

            attention_mask = full.get(
                "attention_mask"
            )

            if attention_mask is not None:
                attention_mask = attention_mask.to(
                    self.device
                )

            prefix_length = (
                prefix["input_ids"].shape[1]
            )

            with torch.no_grad():

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

                logits = outputs.logits

            shift_logits = logits[:, :-1, :]
            shift_labels = input_ids[:, 1:]

            log_probs = torch.log_softmax(
                shift_logits,
                dim=-1,
            )

            token_log_probs = (
                log_probs
                .gather(
                    2,
                    shift_labels.unsqueeze(-1),
                )
                .squeeze(-1)
            )

            start = prefix_length - 1

            target_log_probs = token_log_probs[
                0,
                start:
            ]

            return target_log_probs.sum().item()