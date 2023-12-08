from datasets import load_dataset
import pandas as pd
import mlflow

#  We are now ready to move to the staging step of deployment.  To get started, we will register the model in the MLflow Model Registry (more info below).

# Define the name for the model in the Model Registry.
# We filter out some special characters which cannot be used in model names.
model_name = "summarizer - jacopo"
#model_uri=model_info.model_uri
model_uri = "runs:/83f60f515c7a496daccd42bebaa15fa5/summarizer"
print(model_name)
print(model_uri)

# Register a new model under the given name, or a new model version if the name exists already.
mlflow.register_model(model_uri=model_uri, name=model_name)

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

## Prepare data
xsum_dataset = load_dataset(
    "xsum", version="1.2.0"
)
xsum_sample = xsum_dataset["test"].select(range(10))

results = staging_model.predict(xsum_sample.to_pandas()["document"])
print(pd.DataFrame(results, columns=["generated_summary"]))