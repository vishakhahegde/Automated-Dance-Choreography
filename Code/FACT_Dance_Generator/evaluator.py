
"""Module to evaluate a conditional flow prediction model."""

import os

from absl import app
from absl import flags
from mint.core import inputs
from mint.core import model_builder
from mint.ctl import single_task_evaluator
from mint.utils import config_util
#from third_party.tf_models import orbit
import tensorflow_models as tfm
import orbit
import tensorflow as tf


FLAGS = flags.FLAGS

flags.DEFINE_string('model_dir', '/content/drive/MyDrive/Capstone_Project/mint-main/checkpts_dir',
                    'Directory to write training checkpoints and logs')
flags.DEFINE_string('config_path', '/content/drive/MyDrive/Capstone_Project/mint-main/configs/fact_v5_deeper_t10_cm12.config', 'Path to the config file.')
flags.DEFINE_string('eval_prefix', 'valid', 'Prefix for evaluation summaries.')
flags.DEFINE_string('output_dir', '/content/drive/MyDrive/Capstone_Project/mint-main/output_final2', 'Where to save the results.')

# Unused flags to play nice with xm hyperparameter sweep. Add all flags under
# hyperparameter sweep in trainer.py here.
flags.DEFINE_float('initial_learning_rate', 0.1, 'UNUSED FLAG.')
flags.DEFINE_float('weight_decay', None, 'UNUSED FLAG.')
flags.DEFINE_string('head_initializer', 'he_normal',
                    'Initializer for prediction head.')


def evaluate():
  """Evaluates the given model."""
  configs = config_util.get_configs_from_pipeline_file(FLAGS.config_path)
  model_config = configs['model']
  eval_config = configs['eval_config']
  eval_dataset_config = configs['eval_dataset']
  dataset = inputs.create_input(
      train_eval_config=eval_config,
      dataset_config=eval_dataset_config,
      is_training=False,
      use_tpu=True)

  model_ = model_builder.build(model_config, True)
  model_.global_step = tf.Variable(initial_value=0, dtype=tf.int64)
  metrics_ = model_.get_metrics(eval_config)
  evaluator = single_task_evaluator.SingleTaskEvaluator(
      dataset, model=model_, metrics=metrics_, output_dir=FLAGS.output_dir)

  controller = orbit.Controller(
      evaluator=evaluator,
      checkpoint_manager=tf.train.CheckpointManager(
          tf.train.Checkpoint(model=model_, global_step=model_.global_step),
          directory=FLAGS.model_dir,
          max_to_keep=5),
      eval_summary_dir=os.path.join(FLAGS.model_dir, FLAGS.eval_prefix),
      global_step=model_.global_step)

  controller.evaluate_continuously(timeout=70000)


def main(_):
  evaluate()


if __name__ == '__main__':
  # run Keras in eager mode as well.
  tf.config.experimental_run_functions_eagerly(True)
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

  app.run(main)
