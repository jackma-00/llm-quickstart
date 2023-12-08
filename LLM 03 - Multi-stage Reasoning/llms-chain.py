
#  ### Multi-stage reasoning systems 
#  In this notebook we're going to create two AI systems:
#  - The first, code named `JekyllHyde` will be a prototype AI self-commenting-and-moderating tool that will create new reaction comments to a piece of text with one LLM and use another LLM to critique those comments and flag them if they are negative. To build this we will walk through the steps needed to construct prompts and chains, as well as multiple LLM Chains that take multiple inputs, both from the previous LLM and external. 
#  - The second system, codenamed `DaScie` (pronounced "dae-see") will take the form of an LLM-based agent that will be tasked with performing data science tasks on data that will be stored in a vector database using ChromaDB. We will use LangChain agents as well as the ChromaDB library, as well as the Pandas Dataframe Agent and python REPL (Read-Eval-Print Loop) tool.
#  ----
#  ### ![Dolly](https://files.training.databricks.com/images/llm/dolly_small.png) Learning Objectives
# 
#  By the end of this notebook, you will be able to:
#  1. Build prompt template and create new prompts with different inputs
#  2. Create basic LLM chains to connect prompts and LLMs.
#  3. Construct sequential chains of multiple `LLMChains` to perform multi-stage reasoning analysis. 
#  4. Use langchain agents to build semi-automated systems with an LLM-centric agent to perform internet searches and dataset analysis.

# pip install wikipedia==1.4.0 google-search-results==2.4.2 better-profanity==0.7.0 sqlalchemy==2.0.15

#  ## Generate API tokens
#  For many of the services that we'll using in the notebook, we'll need some API keys. Follow the instructions below to generate your own. 
# 
#  ### Hugging Face Hub
#  1. Go to this [Inference API page](https://huggingface.co/inference-api) and click "Sign Up" on the top right.
# 
#  <img src="https://files.training.databricks.com/images/llm/hf_sign_up.png" width=700>
# 
#  2. Once you have signed up and confirmed your email address, click on your user icon on the top right and click the `Settings` button. 
# 
#  3. Navigate to the `Access Token` tab and copy your token. 
# 
#  <img src="https://files.training.databricks.com/images/llm/hf_token_page.png" width=500>
# 

#  ### SerpApi
# 
#  1. Go to this [page](https://serpapi.com/search-api) and click "Register" on the top right. 
#  <img src="https://files.training.databricks.com/images/llm/serp_register.png" width=800>
# 
#  2. After registration, navigate to your dashboard and `API Key` tab. Copy your API key. 
#  <img src="https://files.training.databricks.com/images/llm/serp_api.png" width=800>
# 
# 

# Copy paste your tokens below

import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "<FILL IN>"
os.environ["SERPAPI_API_KEY"] = "<FILL IN>"

#  ## `JekyllHyde` - A self moderating system for social media
# 
#  In this section we will build an AI system that consists of two LLMs. `Jekyll` will be an LLM designed to read in a social media post and create a new comment. However, `Jekyll` can be moody at times so there will always be a chance that it creates a negative-sentiment comment... we need to make sure we filter those out. Luckily, that is the role of `Hyde`, the other LLM that will watch what `Jekyll` says and flag any negative comments to be removed. 

#  ### Step 1 - Letting Jekyll Speak
#  #### Building the Jekyll Prompt
# 
#  To build `Jekyll` we will need it to be able to read in the social media post and respond as a commenter. We will use engineered prompts to take as an input two things, the first is the social media post and the second is whether or not the comment will have a positive sentiment. We'll use a random number generator to create a chance of the flag to be positive or negative in `Jekyll's` response.


# Let's start with the prompt template

from langchain import PromptTemplate
import numpy as np

# Our template for Jekyll will instruct it on how it should respond, and what variables (using the {text} syntax) it should use.
jekyll_template = """
You are a social media post commenter, you will respond to the following post with a {sentiment} response. 
Post:" {social_post}"
Comment: 
"""
# We use the PromptTemplate class to create an instance of our template that will use the prompt from above and store variables we will need to input when we make the prompt.
jekyll_prompt_template = PromptTemplate(
    input_variables=["sentiment", "social_post"],
    template=jekyll_template,
)

# Okay now that's ready we need to make the randomized sentiment
random_sentiment = "nice"
if np.random.rand() < 0.3:
    random_sentiment = "mean"
# We'll also need our social media post:
social_post = "I can't believe I'm learning about LangChain in this MOOC, there is so much to learn and so far the instructors have been so helpful. I'm having a lot of fun learning! #AI #Databricks"

# Let's create the prompt and print it out, this will be given to the LLM.
jekyll_prompt = jekyll_prompt_template.format(
    sentiment=random_sentiment, social_post=social_post
)
print(f"Jekyll prompt:{jekyll_prompt}")

#  ### Step 2 - Giving Jekyll a brain!
#  ####Building the Jekyll LLM 
# 
#  Note: We provide an option for you to use either Hugging Face or OpenAI. If you continue with Hugging Face, the notebook execution will take a long time (up to 10 mins each cell). If you don't mind using OpenAI, following the next markdown cell for API key generation instructions. 
# 
#  For OpenAI,  we will use their GPT-3 model: `text-babbage-001` as our LLM. 


#  #### OPTIONAL: Use OpenAI's language model
# 
#  If you'd rather use OpenAI, you need to generate an OpenAI key. 
# 
#  Steps:
#  1. You need to [create an account](https://platform.openai.com/signup) on OpenAI. 
#  2. Generate an OpenAI [API key here](https://platform.openai.com/account/api-keys). 
# 
#  Note: OpenAI does not have a free option, but it gives you $5 as credit. Once you have exhausted your $5 credit, you will need to add your payment method. You will be [charged per token usage](https://openai.com/pricing). 
# 
#  **IMPORTANT**: It's crucial that you keep your OpenAI API key to yourself. If others have access to your OpenAI key, they will be able to charge their usage to your account! 

# os.environ["OPENAI_API_KEY"] = "<FILL IN>"

# # To interact with LLMs in LangChain we need the following modules loaded
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
# from langchain.llms import OpenAI

# jekyll_llm = OpenAI(model="text-babbage-001")

## We can also use a model from HuggingFaceHub if we wish to go open-source!

model_id = "EleutherAI/gpt-neo-2.7B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
pipe = pipeline(
    "text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512, device_map='auto'
)
jekyll_llm = HuggingFacePipeline(pipeline=pipe)


#  ### Step 3 - What does Jekyll Say?
#  #### Building our Prompt-LLM Chain
# 
#  We can simplify our input by chaining the prompt template with our LLM so that we can pass the two variables directly to the chain.


from langchain.chains import LLMChain
from better_profanity import profanity


jekyll_chain = LLMChain(
    llm=jekyll_llm,
    prompt=jekyll_prompt_template,
    output_key="jekyll_said",
    verbose=False,
)  # Now that we've chained the LLM and prompt, the output of the formatted prompt will pass directly to the LLM.

# To run our chain we use the .run() command and input our variables as a dict
jekyll_said = jekyll_chain.run(
    {"sentiment": random_sentiment, "social_post": social_post}
)

# Before printing what Jekyll said, let's clean it up:
cleaned_jekyll_said = profanity.censor(jekyll_said)
print(f"Jekyll said:{cleaned_jekyll_said}")


#  ### Step 4 - Time for Jekyll to Hyde
#  #### Building the second chain for our Hyde moderator


#####################################
# 1 We will build the prompt template
# Our template for Hyde will take Jekyll's comment and do some sentiment analysis.
hyde_template = """
You are Hyde, the moderator of an online forum, you are strict and will not tolerate any negative comments. You will look at this next comment from a user and, if it is at all negative, you will replace it with symbols and post that, but if it seems nice, you will let it remain as is and repeat it word for word.
Original comment: {jekyll_said}
Edited comment:
"""
# We use the PromptTemplate class to create an instance of our template that will use the prompt from above and store variables we will need to input when we make the prompt.
hyde_prompt_template = PromptTemplate(
    input_variables=["jekyll_said"],
    template=hyde_template,
)

#####################################
# 2 We connect an LLM for Hyde, (we could use a slightly more advanced model 'text-davinci-003 since we have some more logic in this prompt).

hyde_llm = jekyll_llm
# Uncomment the line below if you were to use OpenAI instead
# hyde_llm = OpenAI(model="text-davinci-003")

#####################################
# 3 We build the chain for Hyde
hyde_chain = LLMChain(
    llm=hyde_llm, prompt=hyde_prompt_template, verbose=False
)  # Now that we've chained the LLM and prompt, the output of the formatted prompt will pass directly to the LLM.

#####################################
# 4 Let's run the chain with what Jekyll last said
# To run our chain we use the .run() command and input our variables as a dict
hyde_says = hyde_chain.run({"jekyll_said": jekyll_said})
# Let's see what hyde said...
print(f"Hyde says: {hyde_says}")


#  ### Step 5 - Creating `JekyllHyde`
#  #### Building our first Sequential Chain


from langchain.chains import SequentialChain

# The SequentialChain class takes in the chains we are linking together, as well as the input variables that will be added to the chain. These input variables can be used at any point in the chain, not just the start.
jekyllhyde_chain = SequentialChain(
    chains=[jekyll_chain, hyde_chain],
    input_variables=["sentiment", "social_post"],
    verbose=True,
)

# We can now run the chain with our randomized sentiment, and the social post!
jekyllhyde_chain.run({"sentiment": random_sentiment, "social_post": social_post})
