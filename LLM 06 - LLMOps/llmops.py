 #  # LLMOps
#  In this example, we will walk through some key steps for taking an LLM-based pipeline to production.  Our pipeline will be familiar to you from previous modules: summarization of news articles using a pre-trained model from Hugging Face.  But in this walkthrough, we will be more rigorous about LLMOps.
# 
#  **Develop an LLM pipeline**
# 
#  Our LLMOps goals during development are (a) to track what we do carefully for later auditing and reproducibility and (b) to package models or pipelines in a format which will make future deployment easier.  Step-by-step, we will:
#  * Load data.
#  * Build an LLM pipeline.
#  * Test applying the pipeline to data, and log queries and results to MLflow Tracking.
#  * Log the pipeline to the MLflow Tracking server as an MLflow Model.
# 
#  **Test the LLM pipeline**
# 
#  Our LLMOps goals during testing (in the staging or QA stage) are (a) to track the LLM's progress through testing and towards production and (b) to do so programmatically to demonstrate the APIs needed for future CI/CD automation.  Step-by-step, we will:
#  * Register the pipeline to the MLflow Model Registry.
#  * Test the pipeline on sample data.
#  * Promote the registered model (pipeline) to production.
# 
#  **Create a production workflow for batch inference**
# 
#  Our LLMOps goals during production are (a) to write scale-out code which can meet scaling demands in the future and (b) to simplify deployment by using MLflow to write model-agnostic deployment code.  Step-by-step, we will:
#  * Load the latest production LLM pipeline from the Model Registry.
#  * Apply the pipeline to an Apache Spark DataFrame.
#  * Append the results to a Delta Lake table.
# 
#  ### Notes about this workflow
# 
#  **This notebook vs. modular scripts**: Since this demo is in a single notebook, we will divide the workflow from development to production via notebook sections.  In a more realistic LLM Ops setup, you would likely have the sections split into separate notebooks or scripts.
# 
#  **Promoting models vs. code**: We track the path from development to production via the MLflow Model Registry.  That is, we are *promoting models* towards production, rather than promoting code.  For more discussion of these two paradigms, see ["The Big Book of MLOps"](https://www.databricks.com/resources/ebook/the-big-book-of-mlops).
# 
#  ### ![Dolly](https://files.training.databricks.com/images/llm/dolly_small.png) Learning Objectives
#  1. Walk through a simple but realistic workflow to take an LLM pipeline from development to production.
#  1. Make use of MLflow Tracking and the Model Registry to package and manage the pipeline.
#  1. Scale out batch inference using Apache Spark and Delta Lake.

#  For this notebook we'll use the <a href="https://huggingface.co/datasets/xsum" target="_blank">Extreme Summarization (XSum) Dataset</a>  with the <a href="https://huggingface.co/t5-small" target="_blank">T5 Text-To-Text Transfer Transformer</a> from Hugging Face.

## Prepare data

from datasets import load_dataset
from transformers import pipeline
from pyspark.sql import SparkSession

# Initialize Spark
spark = SparkSession.builder.getOrCreate()

xsum_dataset = load_dataset(
    "xsum", version="1.2.0"
)
xsum_sample = xsum_dataset["train"].select(range(10))
print(xsum_sample.to_pandas())

#  %md Later on, when we show Production inference, we will want a dataset saved for it.  See the production section below for more information about Delta, the format we use to save the data here.

prod_data_path = "./m6_prod_data"
test_spark_dataset = spark.createDataFrame(xsum_dataset["test"].to_pandas())
test_spark_dataset.write.format("delta").mode("overwrite").save()

## Develop an LLM pipeline

### Create a Hugging Face pipeline

from transformers import pipeline

# Later, we plan to log all of these parameters to MLflow.
# Storing them as variables here will help with that.
hf_model_name = "t5-small"
min_length = 20
max_length = 40
truncation = True
do_sample = True

summarizer = pipeline(
    task="summarization",
    model=hf_model_name,
    min_length=min_length,
    max_length=max_length,
    truncation=truncation,
    do_sample=do_sample,
)

#  We can now examine the `summarizer` pipeline summarizing a document from the `xsum` dataset.

doc0 = xsum_sample["document"][0]
print(f"Summary: {summarizer(doc0)[0]['summary_text']}")
print("===============================================")
print(f"Original Document: {doc0}")

### Track LLM development with MLflow
# 
#  [MLflow](https://mlflow.org/) has a Tracking component that helps you to track exactly how models or pipelines are produced during development.  Although we are not fitting (tuning or training) a model here, we can still make use of tracking to:
#  * Track example queries and responses to the LLM pipeline, for later review or analysis
#  * Store the model as an [MLflow Model flavor](https://mlflow.org/docs/latest/models.html#built-in-model-flavors), thus packaging it for simpler deployment

# Apply to a batch of articles
import pandas as pd

results = summarizer(xsum_sample["document"])
print(pd.DataFrame(results, columns=["summary_text"]))

#  %md [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html) is organized hierarchically as follows:
#  * **An [experiment](https://mlflow.org/docs/latest/tracking.html#organizing-runs-in-experiments)** generally corresponds to the creation of 1 primary model or pipeline.  In our case, this is our LLM pipeline.  It contains some number of *runs*.
#     * **A [run](https://mlflow.org/docs/latest/tracking.html#organizing-runs-in-experiments)** generally corresponds to the creation of 1 sub-model, such as 1 trial during hyperparameter tuning in traditional ML.  In our case, executing this notebook once will only create 1 run, but a second execution of the notebook will create a second run.  This version tracking can be useful during iterative development.  Each run contains some number of logged parameters, metrics, tags, models, artifacts, and other metadata.
#        * **A [parameter](https://mlflow.org/docs/latest/tracking.html#concepts)** is an input to the model or pipeline, such as a regularization parameter in traditional ML or `max_length` for our LLM pipeline.
#        * **A [metric](https://mlflow.org/docs/latest/tracking.html#concepts)** is an output of evaluation, such as accuracy or loss.
#        * **An [artifact](https://mlflow.org/docs/latest/tracking.html#concepts)** is an arbitrary file stored alongside a run's metadata, such as the serialized model itself.
#        * **A [flavor](https://mlflow.org/docs/latest/models.html#storage-format)** is an MLflow format for serializing models.  This format uses the underlying ML library's format (such as PyTorch, TensorFlow, Hugging Face, or your custom format), plus metadata.
# 
#  MLflow has an API for tracking queries and predictions [`mlflow.llm.log_predictions()`](https://mlflow.org/docs/latest/python_api/mlflow.llm.html), which we will use below.  Note that, as of MLflow 2.3.1 (Apr 28, 2023), this API is Experimental, so it may change in later releases.  See the [LLM Tracking page](https://mlflow.org/docs/latest/llm-tracking.html) for more information.
# 
#  ***Tip***: We wrap our model development workflow with a call to `with mlflow.start_run():`.  This context manager syntax starts and ends the MLflow run explicitly, which is a best practice for code which may be moved to production.  See the [API doc](https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.start_run) for more information.

# COMMAND ----------

import mlflow

# Tell MLflow Tracking to use this explicit experiment path,
# which is located on the left hand sidebar under Machine Learning -> Experiments 
mlflow.set_experiment(f"/Users/jacopo/LLM 06 - MLflow experiment")

with mlflow.start_run():
    # LOG PARAMS
    mlflow.log_params(
        {
            "hf_model_name": hf_model_name,
            "min_length": min_length,
            "max_length": max_length,
            "truncation": truncation,
            "do_sample": do_sample,
        }
    )

    # --------------------------------
    # LOG INPUTS (QUERIES) AND OUTPUTS
    # Logged `inputs` are expected to be a list of str, or a list of str->str dicts.
    results_list = [r["summary_text"] for r in results]

    # Our LLM pipeline does not have prompts separate from inputs, so we do not log any prompts.
    mlflow.llm.log_predictions(
        inputs=xsum_sample["document"],
        outputs=results_list,
        prompts=["" for _ in results_list],
    )

    # ---------
    # LOG MODEL
    # We next log our LLM pipeline as an MLflow model.
    # This packages the model with useful metadata, such as the library versions used to create it.
    # This metadata makes it much easier to deploy the model downstream.
    # Under the hood, the model format is simply the ML library's native format (Hugging Face for us), plus metadata.

    # It is valuable to log a "signature" with the model telling MLflow the input and output schema for the model.
    signature = mlflow.models.infer_signature(
        xsum_sample["document"][0],
        mlflow.transformers.generate_signature_output(
            summarizer, xsum_sample["document"][0]
        ),
    )
    print(f"Signature:\n{signature}\n")

    # For mlflow.transformers, if there are inference-time configurations,
    # those need to be saved specially in the log_model call (below).
    # This ensures that the pipeline will use these same configurations when re-loaded.
    inference_config = {
        "min_length": min_length,
        "max_length": max_length,
        "truncation": truncation,
        "do_sample": do_sample,
    }

    # Logging a model returns a handle `model_info` to the model metadata in the tracking server.
    # This `model_info` will be useful later in the notebook to retrieve the logged model.
    model_info = mlflow.transformers.log_model(
        transformers_model=summarizer,
        artifact_path="summarizer",
        task="summarization",
        inference_config=inference_config,
        signature=signature,
        input_example="This is an example of a long news article which this pipeline can summarize for you.",
    )

### Query the MLflow Tracking server
# 
#  **MLflow Tracking API**: We briefly show how to query the logged model and metadata in the MLflow Tracking server, by loading the logged model.  See the [MLflow API](https://mlflow.org/docs/latest/python_api/mlflow.html) for more information about programmatic access.
# 
#  **MLflow Tracking UI**: You can also use the UI.  In the right-hand sidebar, click the beaker icon to access the MLflow experiments run list, and then click through to access the Tracking server UI.  There, you can see the logged metadata and model.  Note in particular that our LLM inputs and outputs have been logged as a CSV file under model artifacts.
# 
#  GIF of MLflow UI:
#  ![GIF of MLflow UI](https://files.training.databricks.com/images/llm/llmops.gif)

#  Now, we can load the pipeline back from MLflow as a [pyfunc](https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html) and use the `.predict()` method to summarize an example document.


loaded_summarizer = mlflow.pyfunc.load_model(model_uri=model_info.model_uri)
loaded_summarizer.predict(xsum_sample["document"][0])

#  The `.predict()` method can handle more than one document at a time, below we pass in all the data from `xsum_sample`.

results = loaded_summarizer.predict(xsum_sample.to_pandas()["document"])
print(pd.DataFrame(results, columns=["generated_summary"]))

#  We are now ready to move to the staging step of deployment.  To get started, we will register the model in the MLflow Model Registry (more info below).

# Define the name for the model in the Model Registry.
# We filter out some special characters which cannot be used in model names.
model_name = "summarizer - jacopo"
model_name = model_name.replace("/", "_").replace(".", "_").replace(":", "_")
print(model_name)

# Register a new model under the given name, or a new model version if the name exists already.
mlflow.register_model(model_uri=model_info.model_uri, name=model_name)

## Test the LLM pipeline
# 
#  During the Staging step of development, our goal is to move code and/or models from Development to Production.  In order to do so, we must test the code and/or models to make sure they are ready for Production.
# 
#  We track our progress here using the [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html).  This metadata and model store organizes models as follows:
#  * **A registered model** is a named model in the registry, in our case corresponding to our summarization model.  It may have multiple *versions*.
#     * **A model version** is an instance of a given model.  As you update your model, you will create new versions.  Each version is designated as being in a particular *stage* of deployment.
#        * **A stage** is a stage of deployment: `None` (development), `Staging`, `Production`, or `Archived`.
# 
#  The model we registered above starts with 1 version in stage `None` (development).
# 
#  In the workflow below, we will programmatically transition the model from development to staging to production.  For more information on the Model Registry API, see the [Model Registry docs](https://mlflow.org/docs/latest/model-registry.html).  Alternatively, you can edit the registry and make model stage transitions via the UI.  To access the UI, click the Experiments menu option in the left-hand sidebar, and search for your model name.

from mlflow import MlflowClient

client = MlflowClient()

client.search_registered_models(filter_string=f"name = '{model_name}'")

#  In the metadata above, you can see that the model is currently in stage `None` (development).  In this workflow, we will run manual tests, but it would be reasonable to run both automated evaluation and human evaluation in practice.  Once tests pass, we will promote the model to stage `Production` to mark it ready for user-facing applications.
# 
#  *Model URIs*: Below, we use model URIs to tell MLflow which model and version we are referring to.  Two common URI patterns for the MLflow Model Registry are:
#  * `f"models:/{model_name}/{model_version}"` to refer to a specific model version by number
#  * `f"models:/{model_name}/{model_stage}"` to refer to the latest model version in a given stage

model_version = 1
dev_model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")
dev_model

#  Note about model dependencies*:
#  When you load the model via MLflow above, you may see warnings about the Python environment.  It is very important to ensure that the environments for development, staging, and production match.
#  * For this demo notebook, everything is done within the same notebook environment, so we do not need to worry about libraries and versions.  However, in the Production section below, we demonstrate how to pass the `env_manager` argument to the method for loading the saved MLflow model, which tells MLflow what tooling to use to recreate the environment.
#  * To create a genuine production job, make sure to install the needed libraries.  MLflow saves these libraries and versions alongside the logged model; see the [MLflow docs on model storage](https://mlflow.org/docs/latest/models.html#storage-format) for more information.  While using Databricks for this course, you can also generate an example inference notebook which includes code for setting up the environment; see [the model inference docs](https://docs.databricks.com/machine-learning/manage-model-lifecycle/index.html#use-model-for-inference) for batch or streaming inference for more information.

#  ### Transition to Staging
# 
#  We will move the model to stage `Staging` to indicate that we are actively testing it.

client.transition_model_version_stage(model_name, model_version, "staging")

staging_model = dev_model

# An actual CI/CD workflow might load the `staging_model` programmatically.  For example:
#   mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{Staging}")
# or
#   mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")

#  %md We now "test" the model manually on sample data. Here, we simply print out results and compare them with the original data.  In a more realistic setting, we might use a set of human evaluators to decide whether the model outperformed the previous model or system.

results = staging_model.predict(xsum_sample.to_pandas()["document"])
print(pd.DataFrame(results, columns=["generated_summary"]))

### Transition to Production
# 
#  The results look great!  :) Let's transition the model to Production.

client.transition_model_version_stage(model_name, model_version, "production")

### Create a production workflow for batch inference
# 
#  Once the LLM pipeline is in Production, it may be used by one or more production jobs or serving endpoints.  Common deployment locations are:
#  * Batch or streaming inference jobs
#  * Model serving endpoints
#  * Edge devices
# 
#  Here, we will show batch inference using Apache Spark DataFrames, with Delta Lake format.  Spark allows simple scale-out inference for high-throughput, low-cost jobs, and Delta allows us to append to and modify inference result tables with ACID transactions.  See the [Apache Spark page](https://spark.apache.org/) and the [Delta Lake page](https://delta.io/) more more information on these technologies.

# Load our data as a Spark DataFrame.
# Recall that we saved this as Delta at the start of the notebook.
# Also note that it has a ground-truth summary column.
prod_data = spark.read.format("delta").load(prod_data_path).limit(10)
print(prod_data)

# Below, we load the model using `mlflow.pyfunc.spark_udf`.  This returns the model as a Spark User Defined Function which can be applied efficiently to big data.  *Note that the deployment code is library-agnostic: it never references that the model is a Hugging Face pipeline.*  This simplified deployment is possible because MLflow logs environment metadata and "knows" how to load the model and run it.

# MLflow lets you grab the latest model version in a given stage.  Here, we grab the latest Production version.
prod_model_udf = mlflow.pyfunc.spark_udf(
    spark,
    model_uri=f"models:/{model_name}/Production",
    env_manager="local",
    result_type="string",
)

# Run inference by appending a new column to the DataFrame

batch_inference_results = prod_data.withColumn(
    "generated_summary", prod_model_udf("document")
)
print(batch_inference_results)

#  %md We can now write out our inference results to another Delta table.  Here, we append the results to an existing table (and create the table if it does not exist).

inference_results_path = "/m6-inference-results"
batch_inference_results.write.format("delta").mode("append").save(
    inference_results_path
)

#  %md And that's it!  To create a production job, we could for example take the new lines of code above, put them in a new notebook, and schedule it as an automated workflow.  MLflow can be integrated with essentially any deployment system, but for more information specific to this Databricks workspace, see the "Use model for inference" documentation for [AWS](https://docs.databricks.com/machine-learning/manage-model-lifecycle/index.html#use-model-for-inference), [Azure](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/manage-model-lifecycle/#--use-model-for-inference), or [GCP](https://docs.gcp.databricks.com/machine-learning/manage-model-lifecycle/index.html#use-model-for-inference).
# 
#  We did not cover model serving for real-time inference, but MLflow models can be deployed to any cloud or on-prem serving systems.  For more information, see the [open-source MLflow Model Registry docs](https://mlflow.org/docs/latest/model-registry.html) or the [Databricks Model Serving docs](https://docs.databricks.com/machine-learning/model-serving/index.html).
# 
#  For other topics not covered, see ["The Big Book of MLOps."](https://www.databricks.com/resources/ebook/the-big-book-of-mlops)

# COMMAND ----------

#  %md ## Summary
# 
#  We have now walked through a full example of going from development to production.  Our LLM pipeline was very simple, but LLM Ops for a more complex workflow (such as fine-tuning a custom model) would be very similar.  You still follow the basic Ops steps of:
#  * Development: Creating the pipeline or model, tracking the process in the MLflow Tracking server and saving the final pipeline or model.
#  * Staging: Registering a new model or version in the MLflow Model Registry, testing it, and promoting it through Staging to Production.
#  * Production: Creating an inference job, or creating a model serving endpoint.

# COMMAND ----------

#  %md-sandbox
#  &copy; 2023 Databricks, Inc. All rights reserved.<br/>
#  Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
#  <br/>
#  <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
