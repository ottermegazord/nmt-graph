

"""
***************************************************************************************
*
*                   Yara English to Cypher Neural Machine Translation
*
*
*  Name : Idaly Ali
*
*  Designation : Research Scientist
*
*  Description : Seq2Seq: Neural Machine Translation with Attention for translation
*                 from English to Cypher - Training
*
*  Reference: TensorFlow 2.0-Alpha Tutorials
*
***************************************************************************************
"""

import util
import tensorflow as tf
import time
import os
import datetime
import argparse
from sklearn.model_selection import train_test_split
from model import Encoder
from model import BahdanauAttention
from model import Decoder

"""Setting up argparse"""
parser = argparse.ArgumentParser(description='Neural Machine Translation with Attention for translation \
                 from English to Cypher')

parser.add_argument("--english_data", type=str,
                    default='data/questions/english.txt', help=".txt file to English questions")
parser.add_argument("--cypher_data", type=str,
                    default='data/questions/cypher.txt', help=".txt file to Cypher answers")
parser.add_argument("--batch_size", type=int,
                    default=64, help="Batch size")
parser.add_argument("--lr", type=float,
                    default=0.001, help="Learning rate")
parser.add_argument("--epochs", type=int,
                    default=4, help="Epochs")
parser.add_argument("--embedding_dim", type=int,
                    default=256, help="Embedding dimensions")
parser.add_argument("--units", type=int,
                    default=1024, help="Units")
parser.add_argument("--checkpoint_dir", type=str,
                    default='./training_checkpoints', help="Directory to checkpoint")
parser.add_argument("--log_dir", type=str,
                    default='logs/gradient_tape', help="Directory to log")


args = parser.parse_args()
# print (args.english_data)
#
"""Paths to Files"""
# Download CSV files containing questions in English and answers in Cypher
ENGLISH_TXT_PATH = args.english_data
CYPHER_TXT_PATH = args.cypher_data


"""Data Pre-processing"""

# Load dataset
input_tensor, target_tensor, inp_lang, targ_lang = util.load_dataset(ENGLISH_TXT_PATH, CYPHER_TXT_PATH)

# Calculate maximum length of target and input tensors
max_length_targ, max_length_inp = util.max_length(target_tensor), util.max_length(input_tensor)

# Create training and validation sets using Pareto method
input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor,
                                                                                                target_tensor,
                                                                                                test_size=0.2)

"""Set Parameters"""

BUFFER_SIZE = len(input_tensor_train)
BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
steps_per_epoch = len(input_tensor_train)//BATCH_SIZE
embedding_dim = args.embedding_dim
units = args.units
vocab_inp_size = len(inp_lang.word_index) + 1
vocab_tar_size = len(targ_lang.word_index) + 1

# Checkpoint Directories
checkpoint_dir = args.checkpoint_dir
log_dir = args.log_dir
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
# checkpoint_dir = './training_checkpoints'

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = log_dir + '/' + current_time + '/train'
test_log_dir = log_dir + '/' + current_time + '/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)

"""Seq2Seq Neural Machine Translation with Attention model"""

# Create TF Dataset
dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

# Create Encoder class
encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)

example_input_batch, example_target_batch = next(iter(dataset))

# sample_hidden = encoder.initialize_hidden_state()
# sample_output, sample_hidden = encoder(example_input_batch, sample_hidden)

attention_layer = BahdanauAttention(10)
# attention_result, attention_weights = attention_layer(sample_hidden, sample_output)

# print(attention_result.shape)
# print(attention_weights.shape)

decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)
# sample_decoder_output, _,_ = decoder(tf.random.uniform((64, 1)),
#                                      sample_hidden, sample_output)


optimizer = tf.keras.optimizers.Adam(lr=args.lr)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

"""Create checkpoint to save model"""
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


@tf.function
def train_step(inp, targ, enc_hidden):
    loss = 0

    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(inp, enc_hidden)

        dec_hidden = enc_hidden

        dec_input = tf.expand_dims([targ_lang.word_index['<start>']] * BATCH_SIZE, 1)

        # Teacher forcing - feeding the target as the next input
        for t in range(1, targ.shape[1]):
            # passing enc_output to the decoder
            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

            loss += loss_function(targ[:, t], predictions)

            # using teacher forcing
            dec_input = tf.expand_dims(targ[:, t], 1)

    batch_loss = (loss / int(targ.shape[1]))

    variables = encoder.trainable_variables + decoder.trainable_variables

    gradients = tape.gradient(loss, variables)

    optimizer.apply_gradients(zip(gradients, variables))

    return batch_loss


"""Training the model"""

for epoch in range(EPOCHS):
    start = time.time()
    enc_hidden = encoder.initialize_hidden_state()
    total_loss = 0

    for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
        batch_loss = train_step(inp, targ, enc_hidden)
        total_loss += batch_loss

        #         tf.summary.scalar('batch_loss', batch_loss.numpy(), step=epoch)

        if batch % 100 == 0:
            print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                         batch,
                                                         batch_loss.numpy()))
    # saving (checkpoint) the model every 2 epochs
    if (epoch + 1) % 1 == 0:
        checkpoint.save(file_prefix=checkpoint_prefix)

    print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                        total_loss / steps_per_epoch))
    print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))