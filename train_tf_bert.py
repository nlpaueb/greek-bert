import logging
import os
import sys
import time
import tensorflow as tf
sys.path.append("bert")
from tensorflow.contrib.cluster_resolver import TPUClusterResolver
from bert import modeling
from bert.run_pretraining import input_fn_builder, model_fn_builder

# configure logging
log = logging.getLogger('tensorflow')
log.setLevel(logging.INFO)

# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s :  %(message)s')
sh = logging.StreamHandler()
sh.setLevel(logging.INFO)
sh.setFormatter(formatter)
log.handlers = [sh]
log.info("Using TPU runtime")
USE_TPU = True
tpu_cluster_resolver = TPUClusterResolver(tpu='greek-bert', zone='us-central1-a')

# SETUP FOLDERS
with tf.Session(tpu_cluster_resolver.get_master()) as session:
    print(tpu_cluster_resolver.get_master())
    HOME_PATH = "gs://greek_bert"  # @param {type:"string"}
    MODEL_DIR = "greek_bert"  # @param {type:"string"}
    PRETRAINING_DIR = "greek_tfrecords"  # @param {type:"string"}
    VOC_FNAME = "vocab.txt"  # @param {type:"string"}

# Input data pipeline config
TRAIN_BATCH_SIZE = 256  # @param {type:"integer"}
MAX_PREDICTIONS =75   # @param {type:"integer"}
MAX_SEQ_LENGTH = 512  # @param {type:"integer"}
MASKED_LM_PROB = 0.15  # @param

# Training procedure config
EVAL_BATCH_SIZE = 256
LEARNING_RATE = 1e-4
TRAIN_STEPS = 1000000  # @param {type:"integer"}
EVAL_STEPS = 10000
SAVE_CHECKPOINTS_STEPS = 50000  # @param {type:"integer"}
NUM_TPU_CORES = 8
BERT_GCS_DIR = "{}/{}".format(HOME_PATH, MODEL_DIR)
DATA_GCS_DIR = "{}/{}".format(HOME_PATH, PRETRAINING_DIR)
VOCAB_FILE = os.path.join(BERT_GCS_DIR, VOC_FNAME)
CONFIG_FILE = os.path.join(BERT_GCS_DIR, "bert_config.json")
INIT_CHECKPOINT = tf.train.latest_checkpoint(BERT_GCS_DIR)
bert_config = modeling.BertConfig.from_json_file(CONFIG_FILE)
input_files = tf.gfile.Glob(os.path.join(DATA_GCS_DIR, '*'))
log.info("Using checkpoint: {}".format(INIT_CHECKPOINT))
log.info("Using {} data shards".format(len(input_files)))
time.sleep(10)

# BUILD TPU ESTIMATOR
model_fn = model_fn_builder(
    bert_config=bert_config,
    init_checkpoint=INIT_CHECKPOINT,
    learning_rate=LEARNING_RATE,
    num_train_steps=TRAIN_STEPS,
    num_warmup_steps=10000,
    use_tpu=USE_TPU,
    use_one_hot_embeddings=True)
run_config = tf.contrib.tpu.RunConfig(
    cluster=tpu_cluster_resolver,
    model_dir=BERT_GCS_DIR,
    save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS,
    tpu_config=tf.contrib.tpu.TPUConfig(
        iterations_per_loop=SAVE_CHECKPOINTS_STEPS,
        num_shards=NUM_TPU_CORES,
        per_host_input_for_training=tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2))
estimator = tf.contrib.tpu.TPUEstimator(
    use_tpu=USE_TPU,
    model_fn=model_fn,
    config=run_config,
    train_batch_size=TRAIN_BATCH_SIZE,
    eval_batch_size=EVAL_BATCH_SIZE)
train_input_fn = input_fn_builder(
    input_files=input_files,
    max_seq_length=MAX_SEQ_LENGTH,
    max_predictions_per_seq=MAX_PREDICTIONS,
    is_training=True)
estimator.train(input_fn=train_input_fn, max_steps=TRAIN_STEPS)
tf.logging.info("***** Running evaluation *****")
tf.logging.info("  Batch size = %d", EVAL_BATCH_SIZE)
eval_input_fn = input_fn_builder(
    input_files=input_files,
    max_seq_length=MAX_SEQ_LENGTH,
    max_predictions_per_seq=MAX_PREDICTIONS,
    is_training=False)
result = estimator.evaluate(
    input_fn=eval_input_fn, steps=EVAL_STEPS)
tf.logging.info("***** Eval results *****")
for key in sorted(result.keys()):
    tf.logging.info("  %s = %s", key, str(result[key]))
