from datasets import load_dataset
from transformers import pipeline

xsum_dataset = load_dataset("xsum", version="1.2.0")

# This dataset provides 3 columns:
#  `document`: the BBC article text
#  `summary`: a "ground-truth" summary --> Note how subjective this "ground-truth" is.  Is this the same summary you would write?  This a great example of how many LLM applications do not have obvious "right" answers.
#  `id`: article ID

xsum_sample = xsum_dataset["train"].select(range(10))
print(xsum_sample.to_pandas())

#  We next use the Hugging Face `pipeline` tool to load a pre-trained model.  In this LLM pipeline constructor, we specify:
#  `task`: This first argument specifies the primary task.  See [Hugging Face tasks](https://huggingface.co/tasks) for more information.
#  `model`: This is the name of the pre-trained model from the [Hugging Face Hub](https://huggingface.co/models).
#  `min_length`, `max_length`: We want our generated summaries to be between these two token lengths.
#  `truncation`: Some input articles may be too long for the LLM to process.  Most LLMs have fixed limits on the length of input sequences.  This option tells the pipeline to truncate the input if needed.

summarizer = pipeline(
    task="summarization",
    model="t5-small",
    min_length=20,
    max_length=40,
    truncation=True,
)

# Apply to 1 article
summarizer(xsum_sample["document"][0])

# Apply to a batch of articles
results = summarizer(xsum_sample["document"])

# Display the generated summary side-by-side with the reference summary and original document.
# We use Pandas to join the inputs and outputs together in a nice format.
import pandas as pd

print(
    pd.DataFrame.from_dict(results)
    .rename({"summary_text": "generated_summary"}, axis=1)
    .join(pd.DataFrame.from_dict(xsum_sample))[
        ["generated_summary", "summary", "document"]
    ]
)