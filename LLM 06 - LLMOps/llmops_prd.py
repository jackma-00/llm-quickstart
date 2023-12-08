from datasets import load_dataset
import pandas as pd
from mlflow import MlflowClient
from pyspark.sql import SparkSession
import mlflow


## Prepare data

# Initialize Spark
spark = SparkSession.builder.appName("MyApp").getOrCreate()


xsum_dataset = load_dataset(
    "xsum", version="1.2.0"
)
#prod_data_path = "./LLM 06 - LLMOps/m6_prod_data"
prod_data = spark.createDataFrame(xsum_dataset["test"].to_pandas())
#test_spark_dataset.write.format("delta").mode("overwrite").save(prod_data_path)

### Transition to Production
# 
#  The results look great!  :) Let's transition the model to Production.

client = MlflowClient()

model_name = "summarizer - jacopo"
model_uri = "runs:/83f60f515c7a496daccd42bebaa15fa5/summarizer"
model_version = 1

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

#  %md And that's it!  To create a production job, we could for example take the new lines of code above, put them in a new notebook, and schedule it as an automated workflow.  MLflow can be integrated with essentially any deployment system, but for more information specific to this Databricks workspace, see the "Use model for inference" documentation for [AWS](https://docs.databricks.com/machine-learning/manage-model-lifecycle/index.html#use-model-for-inference), [Azure](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/manage-model-lifecycle/#--use-model-for-inference), or [GCP](https://docs.gcp.databricks.com/machine-learning/manage-model-lifecycle/index.html#use-model-for-inference).
# 
#  We did not cover model serving for real-time inference, but MLflow models can be deployed to any cloud or on-prem serving systems.  For more information, see the [open-source MLflow Model Registry docs](https://mlflow.org/docs/latest/model-registry.html) or the [Databricks Model Serving docs](https://docs.databricks.com/machine-learning/model-serving/index.html).
# 
#  For other topics not covered, see ["The Big Book of MLOps."](https://www.databricks.com/resources/ebook/the-big-book-of-mlops)