"""An Apache Beam example pipeline to run batch inference jobs using a model trained with AXLearn.
Command line options:
--module: the same module used for training
--config: the same config used for training
--trainer_dir: location of your checkpoints for inference

To debug locally:
$ docker run -it --mount type=bind,src=$HOME/.config/gcloud,dst=/root/.config/gcloud \
    --entrypoint /bin/bash ${DOCKER_REPO}/${DOCKER_IMAGE}:{DOCKER_TAG}
> python3 -m axlearn.cloud.gcp.dataflow_inference_custom \
    --module=text.gpt.c4_trainer \
    --config=fuji-7B-single  \
    --trainer_dir='gs://.../checkpoints/step_xxx'

To use axlearn CLI:
$ axlearn gcp dataflow start \
    --bundler_spec=dockerfile=Dockerfile \
    --bundler_spec=repo=${DOCKER_REPO} \
    --bundler_spec=image=${DOCKER_IMAGE} \
    --bundler_spec=target=dataflow \
    --bundler_spec=allow_dirty=True \
    --dataflow_spec=runner=DirectRunner \
    -- "'python3 -m axlearn.cloud.gcp.dataflow_inference_custom \
        --module=text.gpt.c4_trainer \
        --config=fuji-7B-single \
        --trainer_dir='gs://.../checkpoints/step_xxx' \
        '"

To launch the job locally and run on GCP Dataflow:
$ DOCKER_REPO=
$ DOCKER_IMAGE=
$ axlearn gcp dataflow start \
    --bundler_spec=dockerfile=Dockerfile \
    --bundler_spec=repo=${DOCKER_REPO} \
    --bundler_spec=image=${DOCKER_IMAGE} \
    --bundler_spec=target=dataflow \
    --bundler_spec=allow_dirty=True \
    --dataflow_spec=runner=DataflowRunner \
    -- "'python3 -m axlearn.cloud.gcp.dataflow_inference_custom \
        --module=text.gpt.c4_trainer \
        --config=fuji-7B-single \
        --trainer_dir='gs://.../checkpoints/step_xxx' \
        '"

To use GPUs for your job:
$ DOCKER_REPO=
$ DOCKER_IMAGE=
$ axlearn gcp dataflow start \
    --bundler_spec=dockerfile=Dockerfile \
    --bundler_spec=repo=${DOCKER_REPO} \
    --bundler_spec=image=${DOCKER_IMAGE} \
    --bundler_spec=target=dataflow \
    --bundler_spec=allow_dirty=True \
    --dataflow_spec=runner=DataflowRunner \
    --dataflow_spec=dataflow_service_options=\
    "worker_accelerator=type:nvidia-l4;count:1;install-nvidia-driver" \
    -- "'python3 -m axlearn.cloud.gcp.dataflow_inference_custom \
        --module=text.gpt.c4_trainer \
        --config=fuji-7B-single \
        --trainer_dir='gs://.../checkpoints/step_xxx' \
        '"

"""


import logging
import sys
import inspect
import warnings
from typing import Any, Dict, Optional, Sequence

import apache_beam as beam
import jax
from absl import app, flags
from absl.flags import argparse_flags
from apache_beam.ml.inference.base import ModelHandler, PredictionResult, RunInference
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.internal import pickler as beam_pickler

import axlearn.common.input_fake as input_fake
import axlearn.common.launch_trainer as trainer_utils
from axlearn.common.inference import InferenceRunner, MethodRunner
from axlearn.common.utils import NestedTensor
from axlearn.common.trainer import SpmdTrainer
from axlearn.common.module import Module

import cloudpickle
#import dill as cloudpickle
import dill

from axlearn.experiments.text.gpt import fuji

warnings.filterwarnings("ignore")


class CustomModelHandler(ModelHandler[Dict, PredictionResult, Any]):
    """Defines how to load a model and run inference"""
    def __init__(self, trainer_config: SpmdTrainer.Config, trainer_dir):
        self.trainer_config = trainer_config
        self.trainer_dir = trainer_dir
        logging.info(f"ethan debug __init__2nd check isinstance: {isinstance(trainer_config.model, Module.Config)}")

    def load_model(self) -> MethodRunner:
        """Loads a pre-trained model in the desired type (MethodRunner in this case).
        Reference: https://github.com/apple/axlearn/blob/main/axlearn/common/inference.py#L54

        Returns:
          An instance of MethodRunner.
        """

        # get InferenceRunner Config from Trainer Config and instantiate InferenceRunner
        logging.info(f"Load model, trainer config type:{type(self.trainer_config)}")
        logging.info(f"ethan debug 2nd check isinstance: {isinstance(self.trainer_config.model, Module.Config)}")
        inference_runner_cfg = InferenceRunner.config_from_trainer(self.trainer_config)
        inference_runner_cfg.init_state_builder.set(dir=self.trainer_dir)
        logging.info(f"Load model, inference runner config model type:{type(inference_runner_cfg.model)}")
        logging.info(f"Load model, inference runner config type:{type(inference_runner_cfg)}")
        logging.info(f"ethan debug 3rd check isinstance: {isinstance(inference_runner_cfg.model, Module.Config)}")
        inference_runner = InferenceRunner(cfg=inference_runner_cfg, parent=None)

        # create Method Runner only once
        method_runner = inference_runner.create_method_runner(
            method="predict", prng_key=jax.random.PRNGKey(1)
        )
        return method_runner

    def run_inference(
        self,
        batch: Sequence[NestedTensor],
        model: MethodRunner,
        inference_args: Optional[Dict[str, Any]] = None,
    ):
        """Runs inferences on a batch of NestedTensors.
        NestedTensor: https://github.com/apple/axlearn/blob/main/axlearn/common/utils.py#L56

        Args:
          batch: A sequence of examples as NestedTensors.
          model: An instance of a MethodRunner.
          inference_args: Any additional arguments for an inference.

        Returns:
          A list of type MethodRunner.Output.
        """
        logging.info("RUNNING INFERENCE")
        output_list = []
        for el in batch:
            output_list.append(model(el))

        return output_list


def get_examples() -> Sequence[NestedTensor]:
    """Returns a list of fake input. You can edit this function to return your desired input.
    Fake input: https://github.com/apple/axlearn/blob/main/axlearn/common/input_fake.py#L49

    Returns:
        A list of examples of type FakeLmInput.
        Must be a Sequence since Beam expects a Sequence of examples.
        A Sequence of NestedTensor, Tensor, or other types should all work.
    """
    cfg = input_fake.FakeLmInput.default_config()
    cfg.is_training = False
    cfg.global_batch_size = 1
    cfg.total_num_batches = 1

    fake_input = input_fake.FakeLmInput(cfg)
    example_list = []
    for _ in range(cfg.total_num_batches):
        example_list.append(fake_input.__next__())

    return example_list


def parse_flags(argv):
    """Parse out arguments in addition to the defined absl flags
    (can be found in axlearn/common/launch_trainer.py).
    Addition arguments are returned to the 'main' function by 'app.run'.
    """
    parser = argparse_flags.ArgumentParser(
        description="Parser to parse additional arguments other than defined ABSL flags."
    )
    # Assume all remaining unknown arguments are Dataflow Pipeline options
    _, pipeline_args = parser.parse_known_args(argv[1:])
    return pipeline_args


def main(args, save_main_session=True, pickler="cloudpickle"):
    #FLAGS = flags.FLAGS

    #beam_pickler.set_library(beam_pickler.USE_CLOUDPICKLE)

    # The default pickler is dill and cannot pickle absl FlagValues. Use cloudpickle instead.
    #args.append(f"--pickle_library={pickler}")
    #if save_main_session:
    #    args.append("--save_main_session")
    #args.append("--type_check_additional=all,ptransform_fn")

    # get pipeline input
    #pipeline_input = get_examples()

    # run pipeline
    #pipeline_options = PipelineOptions(args)

    #pipeline = beam.Pipeline(options=pipeline_options)

    #module_config = trainer_utils.get_trainer_config(flag_values=FLAGS)
    module_config = trainer_utils.get_trainer_config()
    #logging.info(f"Main module config type:{type(module_config)}")
    #logging.info(f"Main module config mode type:{type(module_config.model)}")
    #logging.info(f"ethan debug 1st check isinstance: {isinstance(module_config.model, Module.Config)}")

    #dill.detect.badobjects(module_config.model, 1)

    #cp_child_config = cloudpickle.dumps(module_config.model)

    #reload_child_config = cloudpickle.loads(cp_child_config)
    #logging.info(f"ethan debug check isinstance LOCAL reloaded cp_child_config: {isinstance(reload_child_config, Module.Config)}")

    trainer_kwargs = fuji.get_trainer_kwargs("7B", vocab_size=0)

    # model.decoder.transformer.layer.remat_spec['policy'].fn: 'jax._src.ad_checkpoint.save_only_these_names'
    origin_obj = trainer_kwargs["model_cfg"].decoder.transformer.layer.remat_spec.policy#.fn

    #logging.info(f"ethan debug print origin_obj: {origin_obj}")
    logging.info(f"ethan debug print orin_job pickleable: {dill.pickles(origin_obj)}")

    #dill.detect.badobjects(origin_obj)

    #sys.exit(0)

    reloaded_obj = cloudpickle.loads(cloudpickle.dumps(origin_obj))
    logging.info(f"ethan debug print orin_job class: {origin_obj.__class__}")
    logging.info(f"ethan debug print orin_job filepath: {inspect.getfile(origin_obj.__class__)}")

    logging.info(f"ethan debug print orin_job: {id(origin_obj.__class__)}")
    logging.info(f"ethan debug print reloaded_obj: {id(reloaded_obj.__class__)}")
    logging.info(f"ethan debug print reloaded_obj filepath: {inspect.getfile(reloaded_obj.__class__)}")

    #logging.info(f"ethan debug reloaded {id(reloaded_obj.__class__)}: {id(reloaded_obj.__class__)==id(origin_obj.__class__)}")


    #for k, v in reloaded_obj.items():
    #    try:
    #        logging.info(f"Compare {k}")
    #        logging.info(f"ethan debug reloaded {id(v.__class__)}: {id(v.__class__)==id(origin_obj[k].__class__)}")
    #    except Exception as e:
    #        logging.info(f"ethan skipping {e}")

    picked_file = "/tmp/child_config.cloudpickle"

    logging.info(f"ethan debug check origin_job: {id(origin_obj.__class__)}")

    with open(picked_file, "rb") as f:
        try:
            reload_from_file = cloudpickle.load(f)
            logging.info(f"ethan debug check RELOAD-LOCAL-prev-FILE: {id(reload_from_file.__class__)}")
            logging.info(f"ethan debug check RELOAD-LOCAL-prev-FILE EQUAL?: {id(reload_from_file.__class__) == id(origin_obj.__class__)}")
        except Exception as e:
            logging.info(f"EXP SKIP: {e}")

    with open(picked_file, "wb") as f:
        logging.info(f"ethan debug write cloudpickle to {picked_file}")
        cloudpickle.dump(origin_obj, f)

    with open(picked_file, "rb") as f:
        reload_from_file= cloudpickle.load(f)
        logging.info(f"ethan debug check RELOAD-LOCAL-FILE: {id(reload_from_file.__class__)}")

    #with pipeline as p:
    #    (
    #        p
    #        | "CreateInput" >> beam.Create(pipeline_input)
    #        | "RunInference" >> RunInference(CustomModelHandler(trainer_config=module_config, trainer_dir=FLAGS.trainer_dir))
    #        | "PrintOutput" >> beam.Map(print)
    #    )


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    app.run(main, flags_parser=parse_flags)
