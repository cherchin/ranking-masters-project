# Prompts adapted from Eliciting Human Preferences with Language Models (https://arxiv.org/abs/2310.11589v1)

# Generative Active Learning
gen_active_learning = (
    "Your task is to learn the what stocks users like, and would most likely want to be included in their portfolio.",
    "Come up with a potential edge case to learn as much information as you can about what their desired behavior should be under different circumstances. Make sure the edge case addresses different aspects of the system than the edge cases that have already been considered.",
    "An example edge case is: Are you interested in the following stock? Stock Ticker: BTC.",
    # Current cases + Elicitation Transcript
    "Generate the most informative edge case that, when answered, will reveal the most about the desired behavior beyond what has already been queried for above. Generate the edge case in the following format, and nothing else: 'Are you intested in the following stock? Stock Ticker: [STOCK_TICKER]'")

# Generating Questions