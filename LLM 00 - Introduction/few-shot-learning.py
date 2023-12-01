from transformers import pipeline

# Few-shot learning
# In few-shot learning tasks, you give the model an instruction, a few query-response examples of how to follow that instruction, and then a new query.  The model must generate the response for that new query.  This technique has pros and cons: it is very powerful and allows models to be reused for many more applications, but it can be finicky and require significant prompt engineering to get good and reliable results.
# Background reading: See the [Wikipedia page on few-shot learning](https://en.wikipedia.org/wiki/Few-shot_learning_&#40;natural_language_processing&#41;) or [this Hugging Face blog about few-shot learning](https://huggingface.co/blog/few-shot-learning-gpt-neo-and-inference-api).
# In this section, we will use:
# Task: Few-shot learning can be applied to many tasks.  Here, we will do sentiment analysis, which was covered earlier.  However, you will see how few-shot learning allows us to specify custom labels, whereas the previous model was tuned for a specific set of labels.  We will also show other (toy) tasks at the end.  In terms of the Hugging Face `task` specified in the `pipeline` constructor, few-shot learning is handled as a `text-generation` task.
# Data: We use a few examples, including a tweet example from the blog post linked above.
# Model: [gpt-neo-1.3B](https://huggingface.co/EleutherAI/gpt-neo-1.3B), a version of the GPT-Neo model discussed in the blog linked above.  It is a transformer model with 1.3 billion parameters developed by Eleuther AI.  For more details, see the [code on GitHub](https://github.com/EleutherAI/gpt-neo) or the [research paper](https://arxiv.org/abs/2204.06745).

# We will limit the response length for our few-shot learning tasks.
few_shot_pipeline = pipeline(
    task="text-generation",
    model="EleutherAI/gpt-neo-1.3B",
    max_new_tokens=10,
)

# Tip: In the few-shot prompts below, we separate the examples with a special token "###" and use the same token to encourage the LLM to end its output after answering the query.  We will tell the pipeline to use that special token as the end-of-sequence (EOS) token below.

# Get the token ID for "###", which we will use as the EOS token below.
eos_token_id = few_shot_pipeline.tokenizer.encode("###")[0]

# Without any examples, the model output is inconsistent and usually incorrect.
results = few_shot_pipeline(
    """For each tweet, describe its sentiment:

[Tweet]: "This new music video was incredible"
[Sentiment]:""",
    eos_token_id=eos_token_id,
)

print(results[0]["generated_text"])

# With only 1 example, the model may or may not get the answer right.
results = few_shot_pipeline(
    """For each tweet, describe its sentiment:

[Tweet]: "This is the link to the article"
[Sentiment]: Neutral
###
[Tweet]: "This new music video was incredible"
[Sentiment]:""",
    eos_token_id=eos_token_id,
)

print(results[0]["generated_text"])

# With 1 example for each sentiment, the model is more likely to understand!
results = few_shot_pipeline(
    """For each tweet, describe its sentiment:

[Tweet]: "I hate it when my phone battery dies."
[Sentiment]: Negative
###
[Tweet]: "My day has been 👍"
[Sentiment]: Positive
###
[Tweet]: "This is the link to the article"
[Sentiment]: Neutral
###
[Tweet]: "This new music video was incredible"
[Sentiment]:""",
    eos_token_id=eos_token_id,
)

print(results[0]["generated_text"])

#  Just for fun, we show a few more examples below.

# The model isn't ready to serve drinks!
results = few_shot_pipeline(
    """For each food, suggest a good drink pairing:

[food]: tapas
[drink]: wine
###
[food]: pizza
[drink]: soda
###
[food]: jalapenos poppers
[drink]: beer
###
[food]: scone
[drink]:""",
    eos_token_id=eos_token_id,
)

print(results[0]["generated_text"])

# This example sometimes works and sometimes does not, when sampling.  Too abstract?
results = few_shot_pipeline(
    """Given a word describing how someone is feeling, suggest a description of that person.  The description should not include the original word.

[word]: happy
[description]: smiling, laughing, clapping
###
[word]: nervous
[description]: glancing around quickly, sweating, fidgeting
###
[word]: sleepy
[description]: heavy-lidded, slumping, rubbing eyes
###
[word]: confused
[description]:""",
    eos_token_id=eos_token_id,
)

print(results[0]["generated_text"])

# We override max_new_tokens to generate longer answers.
# These book descriptions were taken from their corresponding Wikipedia pages.
results = few_shot_pipeline(
    """Generate a book summary from the title:

[book title]: "Stranger in a Strange Land"
[book description]: "This novel tells the story of Valentine Michael Smith, a human who comes to Earth in early adulthood after being born on the planet Mars and raised by Martians, and explores his interaction with and eventual transformation of Terran culture."
###
[book title]: "The Adventures of Tom Sawyer"
[book description]: "This novel is about a boy growing up along the Mississippi River. It is set in the 1840s in the town of St. Petersburg, which is based on Hannibal, Missouri, where Twain lived as a boy. In the novel, Tom Sawyer has several adventures, often with his friend Huckleberry Finn."
###
[book title]: "Dune"
[book description]: "This novel is set in the distant future amidst a feudal interstellar society in which various noble houses control planetary fiefs. It tells the story of young Paul Atreides, whose family accepts the stewardship of the planet Arrakis. While the planet is an inhospitable and sparsely populated desert wasteland, it is the only source of melange, or spice, a drug that extends life and enhances mental abilities.  The story explores the multilayered interactions of politics, religion, ecology, technology, and human emotion, as the factions of the empire confront each other in a struggle for the control of Arrakis and its spice."
###
[book title]: "Blue Mars"
[book description]:""",
    eos_token_id=eos_token_id,
    max_new_tokens=50,
)

print(results[0]["generated_text"])

# Prompt engineering is a new but critical technique for working with LLMs.  You saw some brief examples above.  As you use more general and powerful models, constructing good prompts becomes ever more important.  Some great resources to learn more are:
# [Wikipedia](https://en.wikipedia.org/wiki/Prompt_engineering) for a brief overview
# [Best practices for prompt engineering with OpenAI API](https://help.openai.com/en/articles/6654000-best-practices-for-prompt-engineering-with-openai-api)
# [🧠 Awesome ChatGPT Prompts](https://github.com/f/awesome-chatgpt-prompts) for fun examples with ChatGPT
