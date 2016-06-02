import tensorflow as tf
def inference(images):
#the image is [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3]
#return :
# Logits.

with tf.variable_scope('conv1') as scope:
  kernel = variable_with_weight_decay(
