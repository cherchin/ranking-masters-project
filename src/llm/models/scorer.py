class Scorer:
    def __init__(self, model):
        self.model = model

    def score_conversation(
        self,
        task,
        conversation,
        gold_response,
    ):

        prompt = f"""<s>
    You are a helpful assistant.

    Original task:
    {task}

    Conversation:
    {conversation}

    Now provide the answer to the original task:
    """

        return self.model.score(
            prompt=prompt,
            target=gold_response,
        )