import os

import numpy as np
import tensorflow as tf

from data_frame import DataFrame
from tf_base_model import TFBaseModel
from tf_utils import (
    time_distributed_dense_layer, temporal_convolution_layer,
    sequence_mean, sequence_smape, shape
)


class DataReader(object):

    def __init__(self, data_dir):
        data_cols = [
            'data',
            'is_nan',
            'page_id',
            'project',
            'access',
            'agent',
            'test_data',
            'test_is_nan'
        ]
        data = [np.load(os.path.join(data_dir, '{}.npy'.format(i))) for i in data_cols]

        self.test_df = DataFrame(columns=data_cols, data=data)
        self.train_df, self.val_df = self.test_df.train_test_split(train_size=0.95)

        print ('train size', len(self.train_df))
        print ('val size', len(self.val_df))
        print ('test size', len(self.test_df))

    def train_batch_generator(self, batch_size):
        return self.batch_generator(
            batch_size=batch_size,
            df=self.train_df,
            shuffle=True,
            num_epochs=10000,
            is_test=False
        )

    def val_batch_generator(self, batch_size):
        return self.batch_generator(
            batch_size=batch_size,
            df=self.val_df,
            shuffle=True,
            num_epochs=10000,
            is_test=False
        )

    def test_batch_generator(self, batch_size):
        return self.batch_generator(
            batch_size=batch_size,
            df=self.test_df,
            shuffle=True,
            num_epochs=1,
            is_test=True
        )

    def batch_generator(self, batch_size, df, shuffle=True, num_epochs=10000, is_test=False):
        batch_gen = df.batch_generator(
            batch_size=batch_size,
            shuffle=shuffle,
            num_epochs=num_epochs,
            allow_smaller_final_batch=is_test
        )
        data_col = 'test_data' if is_test else 'data'
        is_nan_col = 'test_is_nan' if is_test else 'is_nan'
        for batch in batch_gen:
            num_decode_steps = 64
            full_seq_len = batch[data_col].shape[1]
            max_encode_length = full_seq_len - num_decode_steps if not is_test else full_seq_len

            x_encode = np.zeros([len(batch), max_encode_length])
            y_decode = np.zeros([len(batch), num_decode_steps])
            is_nan_encode = np.zeros([len(batch), max_encode_length])
            is_nan_decode = np.zeros([len(batch), num_decode_steps])
            encode_len = np.zeros([len(batch)])
            decode_len = np.zeros([len(batch)])

            for i, (seq, nan_seq) in enumerate(zip(batch[data_col], batch[is_nan_col])):
                rand_len = np.random.randint(max_encode_length - 365 + 1, max_encode_length + 1)
                x_encode_len = max_encode_length if is_test else rand_len
                x_encode[i, :x_encode_len] = seq[:x_encode_len]
                is_nan_encode[i, :x_encode_len] = nan_seq[:x_encode_len]
                encode_len[i] = x_encode_len
                decode_len[i] = num_decode_steps
                if not is_test:
                    y_decode[i, :] = seq[x_encode_len: x_encode_len + num_decode_steps]
                    is_nan_decode[i, :] = nan_seq[x_encode_len: x_encode_len + num_decode_steps]

            batch['x_encode'] = x_encode
            batch['encode_len'] = encode_len
            batch['y_decode'] = y_decode
            batch['decode_len'] = decode_len
            batch['is_nan_encode'] = is_nan_encode
            batch['is_nan_decode'] = is_nan_decode

            yield batch


class cnn(TFBaseModel):

    def __init__(
        self,
        residual_channels=32,
        skip_channels=32,
        dilations=[2**i for i in range(8)]*3,
        filter_widths=[2 for i in range(8)]*3,
        num_decode_steps=64,
        **kwargs
    ):
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        self.dilations = dilations
        self.filter_widths = filter_widths
        self.num_decode_steps = num_decode_steps
        super(cnn, self).__init__(**kwargs)

    def transform(self, x):
        return tf.log(x + 1) - tf.expand_dims(self.log_x_encode_mean, 1)

    def inverse_transform(self, x):
        return tf.exp(x + tf.expand_dims(self.log_x_encode_mean, 1)) - 1

    def get_input_sequences(self):
        self.x_encode = tf.placeholder(tf.float32, [None, None])
        self.encode_len = tf.placeholder(tf.int32, [None])
        self.y_decode = tf.placeholder(tf.float32, [None, self.num_decode_steps])
        self.decode_len = tf.placeholder(tf.int32, [None])
        self.is_nan_encode = tf.placeholder(tf.float32, [None, None])
        self.is_nan_decode = tf.placeholder(tf.float32, [None, self.num_decode_steps])

        self.page_id = tf.placeholder(tf.int32, [None])
        self.project = tf.placeholder(tf.int32, [None])
        self.access = tf.placeholder(tf.int32, [None])
        self.agent = tf.placeholder(tf.int32, [None])

        self.keep_prob = tf.placeholder(tf.float32)
        self.is_training = tf.placeholder(tf.bool)

        # The mean is the average hits for each web sit in all the given dates
        self.log_x_encode_mean = sequence_mean(tf.log(self.x_encode + 1), self.encode_len)
        self.log_x_encode = self.transform(self.x_encode)
        self.x = tf.expand_dims(self.log_x_encode, 2) # minus mean and expand to (batch_size, encode_len, 1)

        # Encoded features, its shape is (batch_size, seq_len, 17)
        self.encode_features = tf.concat([
            tf.expand_dims(self.is_nan_encode, 2),
            tf.expand_dims(tf.cast(tf.equal(self.x_encode, 0.0), tf.float32), 2),
            tf.tile(tf.reshape(self.log_x_encode_mean, (-1, 1, 1)), (1, tf.shape(self.x_encode)[1], 1)),
            tf.tile(tf.expand_dims(tf.one_hot(self.project, 9), 1), (1, tf.shape(self.x_encode)[1], 1)),
            tf.tile(tf.expand_dims(tf.one_hot(self.access, 3), 1), (1, tf.shape(self.x_encode)[1], 1)),
            tf.tile(tf.expand_dims(tf.one_hot(self.agent, 2), 1), (1, tf.shape(self.x_encode)[1], 1)),
        ], axis=2)

        # shape (batch_size, num_decode_steps)
        decode_idx = tf.tile(tf.expand_dims(tf.range(self.num_decode_steps), 0), (tf.shape(self.y_decode)[0], 1))
        # shape (batch_size, num_decode_steps, 64 + 1 + 9 + 3 + 2 = 79)
        self.decode_features = tf.concat([
            tf.one_hot(decode_idx, self.num_decode_steps),
            tf.tile(tf.reshape(self.log_x_encode_mean, (-1, 1, 1)), (1, self.num_decode_steps, 1)),
            tf.tile(tf.expand_dims(tf.one_hot(self.project, 9), 1), (1, self.num_decode_steps, 1)),
            tf.tile(tf.expand_dims(tf.one_hot(self.access, 3), 1), (1, self.num_decode_steps, 1)),
            tf.tile(tf.expand_dims(tf.one_hot(self.agent, 2), 1), (1, self.num_decode_steps, 1)),
        ], axis=2)

        # x is the hits minus mean and transformed by log(1+x) to make it easy to predict
        # shape (batch_size, seq_len, 1)
        return self.x

    #
    # Encode the input x and encoded features using convolution with dilations
    #
    def encode(self, x, features):
        # shape (batch_size, seq_len, 1 + 17 = 18 )
        x = tf.concat([x, features], axis=2)

        # Use tf.einsum to change shape (batch_size, seq_len, 18) to (batch_size, seq_len, residual_channels)
        inputs = time_distributed_dense_layer(
            inputs=x,
            output_units=self.residual_channels,
            activation=tf.nn.tanh,
            scope='x-proj-encode'
        )

        # Use for encoding result
        skip_outputs = []
        # Convolution results based on inputs
        conv_inputs = [inputs]
        for i, (dilation, filter_width) in enumerate(zip(self.dilations, self.filter_widths)):
            # convolution with dilation
            dilated_conv = temporal_convolution_layer(
                inputs=inputs,
                output_units=2*self.residual_channels,
                convolution_width=filter_width,
                causal=True,
                dilation_rate=[dilation],
                scope='dilated-conv-encode-{}'.format(i)
            )
            # split dilated conv into filter and gate, and combine them by multiplying
            conv_filter, conv_gate = tf.split(dilated_conv, 2, axis=2)
            dilated_conv = tf.nn.tanh(conv_filter)*tf.nn.sigmoid(conv_gate)

            # change dilated_conv the shape(batch_size, seq_len, residual_channels) to
            # (batch_size, seq_len, residual_channels + skip_channels)
            outputs = time_distributed_dense_layer(
                inputs=dilated_conv,
                output_units=self.skip_channels + self.residual_channels,
                scope='dilated-conv-proj-encode-{}'.format(i)
            )
            # split into skips and residuals
            skips, residuals = tf.split(outputs, [self.skip_channels, self.residual_channels], axis=2)

            inputs += residuals
            conv_inputs.append(inputs)
            skip_outputs.append(skips)

        # skip_outputs shape (batch_size, seq_len, 32*24=768) -> (batch_size, seq_len, 1)
        skip_outputs = tf.nn.relu(tf.concat(skip_outputs, axis=2))
        h = time_distributed_dense_layer(skip_outputs, 128, scope='dense-encode-1', activation=tf.nn.relu)
        y_hat = time_distributed_dense_layer(h, 1, scope='dense-encode-2')

        # conv_inputs shape(batch_size, seq_len, residual_channel) * 25
        return y_hat, conv_inputs[:-1]

    #
    # Initialize the parameters (weight and bias paramters) for the decode stage using
    # the same variable scopes as in this function.
    #
    # In function decode, it use tf.get_variable to refer the parameters
    #
    # The logic is similar as the function encode. But I think this function
    # merely serves as parameter initialization, has no effect on network graph
    # because it does not use the returned y_hat
    def initialize_decode_params(self, x, features):
        # x shape (batch_size, seq_len , 1)
        # features (batch_size, num_decode_step, 79)
        # after concat, x shape is (batch_size, num_decode_steps, 80)
        x = tf.concat([x, features], axis=2)

        # shape (batch_size, num_decode_step, residual_channels)
        inputs = time_distributed_dense_layer(
            inputs=x,
            output_units=self.residual_channels,
            activation=tf.nn.tanh,
            scope='x-proj-decode'
        )

        skip_outputs = []
        conv_inputs = [inputs]
        for i, (dilation, filter_width) in enumerate(zip(self.dilations, self.filter_widths)):
            # convolution with dilation
            dilated_conv = temporal_convolution_layer(
                inputs=inputs,
                output_units=2*self.residual_channels,
                convolution_width=filter_width,
                causal=True,
                dilation_rate=[dilation],
                scope='dilated-conv-decode-{}'.format(i)
            )
            # split into filter and gate
            conv_filter, conv_gate = tf.split(dilated_conv, 2, axis=2)
            # combine by multiplying
            dilated_conv = tf.nn.tanh(conv_filter)*tf.nn.sigmoid(conv_gate)

            # change shape from (batch_size, num_decode_step, residual_channel) to
            # (batch_size, num_decode_step, residual_channel + skip_channel)
            outputs = time_distributed_dense_layer(
                inputs=dilated_conv,
                output_units=self.skip_channels + self.residual_channels,
                scope='dilated-conv-proj-decode-{}'.format(i)
            )
            # split
            # skips shape (batch_size, num_decode_step, skip_channels)
            # residual shape (batch_size, num_decode_step, residual_channels)
            skips, residuals = tf.split(outputs, [self.skip_channels, self.residual_channels], axis=2)

            inputs += residuals
            conv_inputs.append(inputs)
            skip_outputs.append(skips)

        # Turn skip_outputs into y_hat
        skip_outputs = tf.nn.relu(tf.concat(skip_outputs, axis=2))
        h = time_distributed_dense_layer(skip_outputs, 128, scope='dense-decode-1', activation=tf.nn.relu)
        y_hat = time_distributed_dense_layer(h, 1, scope='dense-decode-2')
        return y_hat

    # x --- encoded infomation for the training steps
    # conv_inputs ----- convolution results for the training steps
    # features ----- decode features for the decode steps.
    def decode(self, x, conv_inputs, features):
        batch_size = tf.shape(x)[0]

        # initialize state tensor arrays
        state_queues = []
        for i, (conv_input, dilation) in enumerate(zip(conv_inputs, self.dilations)):
            batch_idx = tf.range(batch_size)
            # For example, batch_size =5, dilation =3, then
            # batch_idx will be [[0,0,0],[1,1,1],[2,2,2],[3,3,3],[4,4,4]]
            batch_idx = tf.tile(tf.expand_dims(batch_idx, 1), (1, dilation))
            # batch_idx will be [0,0,0,1,1,1,2,2,2,3,3,3,4,4,4]
            batch_idx = tf.reshape(batch_idx, [-1])

            # the begining time step for state queue. the result is a tensor with shape (batch_size,)
            queue_begin_time = self.encode_len - dilation - 1
            # e.g. batch_size=5, dilation=3, the elements in queue_begin_time all are 7, and
            # tf.expand_dims(queue_begin_time,1) will be [[7],[7],[7],[7],[7]]
            # tf.expand_dims(tf.range(dilation),0) will be [[0,1,2]]
            # temporal_idx will be
            #   [[7, 8, 9],
            #    [7, 8, 9],
            #    [7, 8, 9],
            #    [7, 8, 9],
            #    [7, 8, 9]]
            # After reshape, it will be [7,8,9,7,8,9,7,8,9,7,8,9,7,8,9]]
            temporal_idx = tf.expand_dims(queue_begin_time, 1) + tf.expand_dims(tf.range(dilation), 0)
            temporal_idx = tf.reshape(temporal_idx, [-1])

            # idx is used as argument of tf.gather to retrieve elements in conv_input.
            # idx has shape (batch_size * dilation,2)
            idx = tf.stack([batch_idx, temporal_idx], axis=1)
            # slice from conv_input with dilation for each batch, shape(batch_size, dilation, feature_size=32)
            slices = tf.reshape(tf.gather_nd(conv_input, idx), (batch_size, dilation, shape(conv_input, 2)))

            layer_ta = tf.TensorArray(dtype=tf.float32, size=dilation + self.num_decode_steps)
            # (batch_size, dilation, feature_size)
            layer_ta = layer_ta.unstack(tf.transpose(slices, (1, 0, 2)))
            state_queues.append(layer_ta)

        # initialize feature tensor array
        # features shape (batch_size, num_decode_steps, 64 + 1 + 9 + 3 + 2 = 79)
        # after shaped, it will be num_decode_steps * (batch_size, 79)
        features_ta = tf.TensorArray(dtype=tf.float32, size=self.num_decode_steps)
        features_ta = features_ta.unstack(tf.transpose(features, (1, 0, 2)))

        # initialize output tensor array
        emit_ta = tf.TensorArray(size=self.num_decode_steps, dtype=tf.float32)

        # initialize other loop vars
        elements_finished = 0 >= self.decode_len
        time = tf.constant(0, dtype=tf.int32)

        # get initial x input
        # idx for x, like (e.g batch_size=32, encode_len=366)
        #   [[0,365],[1,365],...[31,365]])
        # here x is the return y_hat of function encode, with shape (batch_size, seq_len, 1)
        current_idx = tf.stack([tf.range(tf.shape(self.encode_len)[0]), self.encode_len - 1], axis=1)
        initial_input = tf.gather_nd(x, current_idx) #shape (batch_size, 1)

        # current_input is the encoded input at last encode step (i.e. initial_input)
        # or last decode step (i.e. the variable y_hat at the end of loop_fn)
        # queues are the convolution results with dilation defined above in the variable state_queues
        def loop_fn(time, current_input, queues):
            # read the decode features for the time-th decode step as current features
            current_features = features_ta.read(time)
            # current input is the encoded result for the last step. Initial_input is
            # the encoded info at the last encode_step
            # concat input and features as the input for the next steps
            # current_input has shape (batch_size,1),
            # current_features shape (batch_size,79)
            # after concat, it will be (batch_size, 80)
            current_input = tf.concat([current_input, current_features], axis=1)

            # use the variables initialzed in scope x-proj-decode of the initialize_decode_params function
            # x_proj shape (batch_size, 32)
            with tf.variable_scope('x-proj-decode', reuse=True):
                w_x_proj = tf.get_variable('weights') #shape (input.shape[2]+feature.shape[2]=80,residual_channels=32)
                b_x_proj = tf.get_variable('biases') # shape(32,)
                x_proj = tf.nn.tanh(tf.matmul(current_input, w_x_proj) + b_x_proj)

            skip_outputs, updated_queues = [], []
            for i, (conv_input, queue, dilation) in enumerate(zip(conv_inputs, queues, self.dilations)):
                # read state at the time-th state which is the sliced conv_input information
                state = queue.read(time) # queue shape(dilation, batch_size, feature_size)
                with tf.variable_scope('dilated-conv-decode-{}'.format(i), reuse=True):
                    w_conv = tf.get_variable('weights'.format(i)) # shape (conv_width=2, residual_channel=32, residual_channels + skip_channels)
                    b_conv = tf.get_variable('biases'.format(i))
                    dilated_conv = tf.matmul(state, w_conv[0, :, :]) + tf.matmul(x_proj, w_conv[1, :, :]) + b_conv
                conv_filter, conv_gate = tf.split(dilated_conv, 2, axis=1)
                dilated_conv = tf.nn.tanh(conv_filter)*tf.nn.sigmoid(conv_gate)

                with tf.variable_scope('dilated-conv-proj-decode-{}'.format(i), reuse=True):
                    w_proj = tf.get_variable('weights'.format(i)) #shape (residual_channels, residual_channels + skip_channels)
                    b_proj = tf.get_variable('biases'.format(i))
                    concat_outputs = tf.matmul(dilated_conv, w_proj) + b_proj
                skips, residuals = tf.split(concat_outputs, [self.skip_channels, self.residual_channels], axis=1)

                x_proj += residuals # shape (batch_size, residule_channels)
                skip_outputs.append(skips)
                updated_queues.append(queue.write(time + dilation, x_proj))

            skip_outputs = tf.nn.relu(tf.concat(skip_outputs, axis=1))
            with tf.variable_scope('dense-decode-1', reuse=True):
                w_h = tf.get_variable('weights') #  shape (24*32=768, 128)
                b_h = tf.get_variable('biases')
                h = tf.nn.relu(tf.matmul(skip_outputs, w_h) + b_h)

            with tf.variable_scope('dense-decode-2', reuse=True):
                w_y = tf.get_variable('weights')
                b_y = tf.get_variable('biases')
                y_hat = tf.matmul(h, w_y) + b_y

            elements_finished = (time >= self.decode_len)
            finished = tf.reduce_all(elements_finished)

            next_input = tf.cond(
                finished,
                lambda: tf.zeros([batch_size, 1], dtype=tf.float32),
                lambda: y_hat
            )
            next_elements_finished = (time >= self.decode_len - 1)

            return (next_elements_finished, next_input, updated_queues)

        def condition(unused_time, elements_finished, *_):
            # not all the elements are finished, continue the loop
            return tf.logical_not(tf.reduce_all(elements_finished))

        # loop body
        def body(time, elements_finished, emit_ta, *state_queues):
            # call loop_fn to do the real loop logic
            (next_finished, emit_output, state_queues) = loop_fn(time, initial_input, state_queues)

            # why set zero when finished???
            emit = tf.where(elements_finished, tf.zeros_like(emit_output), emit_output)
            # write emit as the result for the time-th step in emit_ta
            emit_ta = emit_ta.write(time, emit)

            # if elements_finished or next_finished is true.
            elements_finished = tf.logical_or(elements_finished, next_finished)
            return [time + 1, elements_finished, emit_ta] + list(state_queues)

        # dynamic loop in tensor
        returned = tf.while_loop(
            cond=condition,
            body=body,
            loop_vars=[time, elements_finished, emit_ta] + state_queues
        )

        # outputs_ta is equal to previous emit_ta
        outputs_ta = returned[2] # shape (batch_size, num_decode_step, 1)
        y_hat = tf.transpose(outputs_ta.stack(), (1, 0, 2))
        return y_hat

    def calculate_loss(self):
        x = self.get_input_sequences()

        # y_hat_encode shape (batch_size, seq_len, 1)
        # conv_inputs shape 24 * (batch_size, seq_len, residual_channels)
        # x shape( batch_size, ?, 1)
        y_hat_encode, conv_inputs = self.encode(x, features=self.encode_features)
        self.initialize_decode_params(x, features=self.decode_features)
        y_hat_decode = self.decode(y_hat_encode, conv_inputs, features=self.decode_features)
        y_hat_decode = self.inverse_transform(tf.squeeze(y_hat_decode, 2))
        y_hat_decode = tf.nn.relu(y_hat_decode)

        self.labels = self.y_decode
        self.preds = y_hat_decode
        self.loss = sequence_smape(self.labels, self.preds, self.decode_len, self.is_nan_decode)

        self.prediction_tensors = {
            'priors': self.x_encode,
            'labels': self.labels,
            'preds': self.preds,
            'page_id': self.page_id,
        }

        return self.loss


if __name__ == '__main__':
    base_dir = './'

    dr = DataReader(data_dir=os.path.join(base_dir, 'data/processed/'))

    nn = cnn(
        reader=dr,
        log_dir=os.path.join(base_dir, 'logs'),
        checkpoint_dir=os.path.join(base_dir, 'checkpoints'),
        prediction_dir=os.path.join(base_dir, 'predictions'),
        optimizer='adam',
        learning_rate=.001,
        batch_size=128,
        num_training_steps=200000,
        early_stopping_steps=5000,
        warm_start_init_step=0,
        regularization_constant=0.0,
        keep_prob=1.0,
        enable_parameter_averaging=False,
        num_restarts=2,
        min_steps_to_checkpoint=500,
        log_interval=10,
        num_validation_batches=1,
        grad_clip=20,
        residual_channels=32,
        skip_channels=32,
        dilations=[2**i for i in range(8)]*3,
        filter_widths=[2 for i in range(8)]*3,
        num_decode_steps=64,
    )
    nn.fit()
    nn.restore()
    nn.predict()
