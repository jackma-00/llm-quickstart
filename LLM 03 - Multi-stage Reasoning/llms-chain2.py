#  ## `DaScie` - Our first vector database data science AI agent!
# 
#  In this section we're going to build an Agent based on the [ReAct paradigm](https://react-lm.github.io/) (or thought-action-observation loop) that will take instructions in plain text and perform data science analysis on data that we've stored in a vector database. The agent type we'll use is using zero-shot learning, which takes in the prompt and leverages the underlying LLMs' zero-shot abilities. 


#  ### Step 1 - Hello DaScie! 
#  #### Creating a data science-ready agent with LangChain!
# 
#  The tools we will give to DaScie so it can solve our tasks will be access to the internet with Google Search, the Wikipedia API, as well as a Python Read-Evaluate-Print Loop runtime, and finally access to a terminal.
# 


# For DaScie we need to load in some tools for it to use, as well as an LLM for the brain/reasoning
from langchain.agents import load_tools  # This will allow us to load tools we need
from langchain.agents import initialize_agent
from langchain.agents import (
    AgentType,
)  # We will be using the type: ZERO_SHOT_REACT_DESCRIPTION which is standard
from langchain.llms import OpenAI

# if use Hugging Face
# llm = jekyll_llm

# For OpenAI we'll use the default model for DaScie
llm = OpenAI()
tools = load_tools(["wikipedia", "serpapi", "python_repl", "terminal"], llm=llm)
# We now create DaScie using the "initialize_agent" command.
dascie = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

 
#  ### Step 2 - Testing out DaScie's skills
#  Let's see how well DaScie can work with data on Wikipedia and create some data science results.


dascie.run(
    "Create a dataset (DO NOT try to download one, you MUST create one based on what you find) on the performance of the Mercedes AMG F1 team in 2020 and do some analysis. You need to plot your results."
)


# Let's try to improve on these results with a more detailed prompt.
dascie.run(
    "Create a detailed dataset (DO NOT try to download one, you MUST create one based on what you find) on the performance of each driver in the Mercedes AMG F1 team in 2020 and do some analysis with at least 3 plots, use a subplot for each graph so they can be shown at the same time, use seaborn to plot the graphs."
)

 
#  ### Step 3 - Using some local data for DaScie.
#  Now we will use some local data for DaScie to analyze.
# 
# 
#  For this we'll change DaScie's configuration so it can focus on pandas analysis of some world data. Source: https://www.kaggle.com/datasets/arnabchaki/data-science-salaries-2023


from langchain.agents import create_pandas_dataframe_agent
import pandas as pd

datasci_data_df = pd.read_csv(f"{DA.paths.datasets}/salaries/ds_salaries.csv")
# world_data
dascie = create_pandas_dataframe_agent(
    OpenAI(temperature=0), datasci_data_df, verbose=True
)


# Let's see how well DaScie does on a simple request.
dascie.run("Analyze this data, tell me any interesting trends. Make some pretty plots.")


# Not bad! Now for something even more complex.... can we get out LLM model do some ML!?
dascie.run(
    "Train a random forest regressor to predict salary using the most important features. Show me the what variables are most influential to this model"
)


#  &copy; 2023 Databricks, Inc. All rights reserved.<br/>
#  Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
#  <br/>
#  <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
