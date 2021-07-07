
from kfp import dsl
import mlrun
from mlrun.platforms import auto_mount

funcs = {}
data_source_url = 'store:///raw_data' 
labels = 'label'
GPUS = False

cleaned_dataset = 'cleaned_data'
model = 'logisticRegression_model'


# Configure function resources and local settings
def init_functions(functions: dict, project=None, secrets=None):
    for f in functions.values():
        f.apply(auto_mount())

# Create a Kubeflow Pipelines pipeline
@dsl.pipeline(
      name="SecureNet",
    description="The workflow implements data processing, training and testing the model and serving the model"
)
def kfpipeline():

    # Ingest the data set
    ingest = funcs['data-clean'].as_step(
    name='data_clean',
    handler='data_clean',
    inputs={"src": data_source_url,'cleaned_key':cleaned_dataset},
    outputs=[cleaned_dataset])
    
    #analyze data
    
    describe = funcs['describe'].as_step(
    name="summary",
    params={"label_column": labels},
    inputs={"table": ingest.outputs[cleaned_dataset]})
    
    # Train a model   
    train = funcs["train-data"].as_step(
        name="train_data",
        params={"label_column": labels},
        inputs={"dataset": ingest.outputs[cleaned_dataset]},
        outputs=['model', 'test_set'])
    

    test = funcs["test"].as_step(
        name="test",
        params={"label_column": labels},
        inputs={"models_path": train.outputs['model'],
                "test_set": train.outputs['test_set']})
    
    # Deploy the model as a serverless function
    deploy = funcs["serving"].deploy_step(
        models={f"{model}_v1": train.outputs['model']})
