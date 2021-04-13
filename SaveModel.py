from ActorCritic import ActorCritic
import numpy as np
import tensorflow as tf

def main():
    model = ActorCritic()
    input_shape = (1, 6, 7, 1)
    state = tf.convert_to_tensor(np.array(
        [[-0., -0., -0., -0., -0., -0., -0.],
        [-0., -0., -0., -0., -0., -0., -0.],
        [ 1., -0., -0., -0.,  1., -0., -0.],
        [-1., -0., -0., -0.,  1.,  1., -0.],
        [-1., -0., -0., 0, -1., -1., -1.],
        [ 1.,  1.,  1., -1.,  1., -1., -1.]]).reshape(1,6,7,1),
        dtype=tf.float32
        )
    _ = model(state)
    model.save('../my_model')

main()