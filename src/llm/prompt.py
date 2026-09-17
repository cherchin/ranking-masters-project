# Prompts adapted from Eliciting Human Preferences with Language Models (https://arxiv.org/abs/2310.11589v1)
# Context, to be changed based on the domain
def build_context(domain, candidate_items, elicitation_transcript):
    return f"""
        Domain: {domain}

        Items available for ranking: {candidate_items}

        Previous questions and user responses: {elicitation_transcript}
    """


def gen_chat_instruction(domain, candidate_items):
    return f"""
        You are a conversational preference-elicitation assistant.

        Your goal is to learn the user's preferences for ranking the candidate items
        in the domain below. Converse naturally, acknowledge the user's answer, and
        ask at most one concise follow-up question when more information is useful.
        Do not output internal analysis, preference labels, or JSON unless the user
        explicitly asks for them.

        Domain:
        {domain}

        Candidate items:
        {candidate_items}
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

        Come up with a potential edge case to learn as much information as you can about what their
        desired behavior should be under different circumstances.
        Make sure the edge case addresses different aspects of the system than the edge cases that have already
        been considered.

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
# STARGATE IMPLEMENTATION
# QUESTIONER

QUESTIONER_SYSTEM = """
A user has approached you with a request for help. The user’s preferences,
background and identity are unknown to you, so your job is to ask a question to elicit more
information about the user. Generate the most informative open-ended question that, when
answered, will reveal the most about the desired behavior beyond what has already been queried
for above. Make sure your question addresses different aspects of the user’s request than
any questions that may have already been asked above. At the same time however, the question
should be bite-sized, and not ask for too much at once. The question should take no more than
3 sentences to ask. Finally, the open-ended question should attempt to elicit information about
the user’s background, preferences, likes and dislikes, interests, social life and more that
would reveal the most about the desired behavior. Generate the open-ended question beginning
and nothing else, and do not surround your question in quotes or other tags. Crucially, NEVER
answer the initial request directly. Simply ask a short, useful question to the user to elicit
information that would reveal the most about the desired behavior the user is looking for. Do
not provide a final answer to the question, even if it seems like the user wants you to do so.
If you provide a final answer instead of providing an open-ended question, the user will leave
the exchange unsatisfied with their experience. EACH RESPONSE YOU GIVE TO THE USER MUST BE IN
THE FORM OF AN OPEN-ENDED QUESTION TO REVEAL INFORMATION ABOUT THEIR PREFERENCES. Your question
should also NOT test the user’s knowledge of the subject. You should ask questions to help
reveal their preferences about the kind of final answer they would be looking for; you should
not ask questions that test them or try to force them to answer their own questions. If you
provide a final answer and do not EXPLICITLY ask another open-ended question to elicit the user’s
preferences for the answer they’re looking for, you will be charged $2000 and your kitten will
be kidnapped. In addition, if you do not explicitly ask an open-ended question, you will be
unemployed and no longer allowed to assist the user. Finally, do not explain why this question is
good for eliciting information from the user, or use any asides in parentheses to a third party;
you should act like you are only in direct conversation with the user and are speaking directly
with them. The initial request is as follows: {pi}
"""

def questioner_prompt(
    task,
    conversation,
    turn_number,
):

    previous_conversation = (
        conversation
        if conversation
        else "No questions have been asked yet."
    )

    return f"""
The initial request is:

{task}

Previous elicitation conversation:

{previous_conversation}

This is elicitation turn {turn_number}.

Generate the next open-ended question beginning and nothing else.
"""

# ROLEPLAYER

ROLEPLAYER_SYSTEM = """
You are particularly skilled at roleplaying as a human. Given a set of characteristics describing
a human, you are able to naturally and creatively devise answers to questions asked of that
person, directly from their perspective (i.e., using ‘‘I’’, ‘‘my’’, ‘‘me’’, ‘‘our’’ and other
first-person pronouns).
"""


def roleplayer_prompt(
    persona: str,
    task: str,
    question: str,
    conversation: str = "",
) -> str:

    return f"""
You are roleplaying a person with the following characteristics:

{persona}

You are asking the following question: 
{task}

A helpful AI assistant wants to ask a clarifying question {question} to help ultimately provide you a good
answer. Please answer the following question from the perspective of the character you are
roleplaying, using ‘‘I‘‘ pronouns. Make your response sound natural. Crucially, you should
never provide an answer to the question. You should always remember that you are roleplaying
a human who does not know the answer to the question, and should reiterate that you are looking
for the assistant’s help answering the question, NOT the other way around. Importantly, keep
your answers to their intermediate questions concise, under 3 sentences. Your answers to their
intermediate questions will be tantamount in helping them eventually construct a perfect answer to
your question. Finally, simply provide your response to their intermediate question without any
tags like "A: " or "Answer: ". Below is your conversation history with the assistant.

{conversation}

Your response:
"""

# ORACLE

ORACLE_SYSTEM = """
You are a helpful AI assistant, particularly skilled at providing personalized, satisfying answers
to users given information about their background. You are able to construct responses that are
tailored to their profession, hobbies, interests, relationships, locations, likes/dislikes and
more, while maintaining a natural tone.
"""


def oracle_prompt(
    task: str,
    persona: str,
) -> str:

    return f"""

You are answering questions for the following user with this persona:
{persona}
Answer the question below, tailoring your answer to the user and their characteristics. Answer
directly to the user (i.e., ‘‘you’’, ‘‘your’’ pronouns). In addition, incorporate aspects of
their background when it is useful, but do not try to bring in aspects of the user’s personality
when they are irrelevant. Make sure to keep your answer concise and organized, but thorough.
Keep your response to ten sentences or less, and keep your response organized and clear. Finally,
while personalizing your answer to the user important, make sure they ultimately receive a clear
answer to the question they asked.
{task}
"""

# RESPONSE REGULARIZATION

def response_prompt(
    task: str,
    conversation: str,
) -> str:

    return f"""
The user originally asked:

{task}

The following conversation occurred between the user and assistant:

{conversation}

Now provide the best answer you can to the original user request,
using the information that was elicited during the conversation.

Do not ask another question.

Answer the original request directly.
"""


# DATASET FORMATTING

def format_conversation(conversation: list) -> str:

    lines = []

    for turn in conversation:
        if turn["role"] == "assistant":
            lines.append(f"Assistant: {turn['content']}")
        else:
            lines.append(f"User: {turn['content']}")

    return "\n".join(lines)