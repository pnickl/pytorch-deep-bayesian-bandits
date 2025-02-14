from bandits.data.mnist import get_raw_features, get_vae_features, construct_dataset_from_features
from bandits.algorithms.neural_linear_sampling import NeuralLinearPosteriorSampling
from bandits.core.hyperparams import HyperParams
from bandits.core.contextual_bandit import run_contextual_bandit
import numpy as np
import sys

np.set_printoptions(threshold=sys.maxsize)

def main():
    data_type = 'mnist'

    vae_data = get_vae_features()
    features, rewards, opt_vals = construct_dataset_from_features(vae_data)
    dataset = np.hstack((features, rewards))

    context_dim = features.shape[1]
    num_actions = 10


    init_lrs = [0.001, 0.0025, 0.005, 0.01]
    base_lrs = [0.0005, 0.001]
    modes = ["triangulbayesar", "triangular2", "exp_range"]
    batch_sizes = [32, 128, 512]
    layer_sizes = [[50, 50], [100, 100], [100]]
    # hyperparams
    for init_lr in init_lrs:
        for base_lrs in base_lrs:
            for mode in modes:
                hp_nlinear = HyperParams(num_actions=num_actions,
                                         context_dim=context_dim,
                                         init_scale=0.3,
                                         layer_sizes=[50, 50],
                                         batch_size=32,
                                         activate_decay=True,
                                         initial_lr=0.1,
                                         max_grad_norm=5.0,
                                         show_training=False,
                                         freq_summary=1000,
                                         buffer_s=-1,
                                         initial_pulls=2,
                                         reset_lr=True,
                                         base_lr=0.01,
                                         lr_decay_rate=0.5,
                                         update_freq_post=1,
                                         training_freq_network=50,
                                         training_epochs=100,
                                         a0=6,
                                         b0=6,
                                         lambda_prior=0.25,
                                         keep_prob=1.0,
                                         global_step=1,
                                         mode=mode)

    algos = [NeuralLinearPosteriorSampling('NeuralLinear', hp_nlinear)]

    # run contextual bandit experiment
    print(context_dim, num_actions)
    results = run_contextual_bandit(context_dim, num_actions, dataset, algos)
    actions, rewards = results
    np.save("results.npy", rewards)


if __name__ == '__main__':
    main()
