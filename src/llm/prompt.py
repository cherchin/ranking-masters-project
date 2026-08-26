# Prompts adapted from Eliciting Human Preferences with Language Models (https://arxiv.org/abs/2310.11589v1)
# Context, to be changed based on the domain
def build_context(domain, candidate_items, elicitation_transcript):
    return f"""
        Domain: {domain}

        Items available for ranking: {candidate_items}

        Previous questions and user responses: {elicitation_transcript}
    """

# Generative Active Learning
def gen_active_learning(domain, candidate_items, elicitation_transcript):
    return f"""
        Your task is to learn the user's preferences in order to improve how a set of
        candidate items should be ranked.

        Given the items available, the user's previous preference responses, and the
        elicitation transcript, generate the single most informative next preference
        query.

        Previous questions and user responses:
        {elicitation_transcript}

        Candidate items:
        {candidate_items}

        Domain:
        {domain}

        The query should help distinguish between plausible user preferences,
        priorities, trade-offs, or ranking behavior that cannot yet be inferred from
        the existing information.

        Consider edge cases, conflicts between ranking criteria, and cases where the
        user's preference may change under different circumstances.

        Do not repeat information that has already been elicited. Prefer a query that
        provides information about aspects of the user's ranking preferences that are
        currently uncertain.

        Generate only the preference query and nothing else.
    """

# Generating Questions
def gen_questions(domain, candidate_items, elicitation_transcript):
    return f"""
        Your task is to learn a user's preferences for ranking a set of candidate items.

        Seek to understand the user's preferences broadly rather than making assumptions based on their previous
        answers. Do not assume that a user has provided a complete description of their
        preferences, so continue probing relevant but unexplored aspects of what influences
        how they would prefer items to be ranked.

        Previous questions and user responses:
        {elicitation_transcript}

        Candidate items:
        {candidate_items}

        Domain:
        {domain}

        Generate the single most informative preference question that, when answered, will
        reveal the most about the user's desired ranking behavior beyond what has already
        been elicited.

        The question should explore an aspect of the user's preferences that has not yet
        been sufficiently examined. Avoid repeating or asking about the same preference
        dimension as previous questions unless further clarification is necessary.

        The question should be concise and bite-sized, asking about only one meaningful
        preference, trade-off, or decision at a time.

        Phrase the question in a way that is understandable to non-expert users. Do not use
        jargon without explanation.

        Generate only the question and nothing else.
    """

# To assess whether the user's ranking preferences are primarily attribute-based or item-based
def identify_preference_type(domain, candidate_items, elicitation_transcript):
    return f"""
        Your task is to determine whether a user's ranking preferences are primarily
        attribute-based or item-based.

        Definitions:

        Attribute-based preferences:
        The user expresses preferences based primarily on characteristics, features,
        properties, or attributes of items. These preferences could potentially be
        generalized to other items with similar attributes.

        Item-based preferences:
        The user expresses preferences primarily for or against specific individual
        items, rather than general characteristics shared by multiple items.

        Domain:
        {domain}

        Candidate items:
        {candidate_items}

        Elicitation transcript:
        {elicitation_transcript}

        Analyze the user's responses in the elicitation transcript.

        Determine which type of preference is more strongly supported by the evidence:

        ATTRIBUTE_BASED
        The user's preferences are primarily explained by general item attributes,
        features, or characteristics.

        ITEM_BASED
        The user's preferences are primarily explained by preferences for specific
        individual items.

        If there is insufficient evidence to determine either preference type, return:

        UNCERTAIN

        Generate only one of the following labels and nothing else:

        ATTRIBUTE_BASED
        ITEM_BASED
        UNCERTAIN
    """