import matplotlib.pyplot as plt
import torch

from vp_diffused_priors import get_vpdiff_gaussian_score


if __name__ == "__main__":
    from tasks.toy_examples.data_generators import SBIGaussian2d
    from nse import NSE, NSELoss
    from sm_utils import train


    torch.manual_seed(1)

    N_TRAIN = 10_000
    N_SAMPLES = 4096
    TIME_POINT = 0.001

    # Task
    task = SBIGaussian2d(prior_type="gaussian")
    # Prior and Simulator
    prior = task.prior
    simulator = task.simulator

    # Train data
    theta_train = task.prior.sample((N_TRAIN,))
    x_train = simulator(theta_train)

    # normalize theta
    theta_train_ = (theta_train - theta_train.mean(axis=0)) / theta_train.std(axis=0)

    # normalize x
    x_train_ = (x_train - x_train.mean(axis=0)) / x_train.std(axis=0)

    # score_network
    dataset = torch.utils.data.TensorDataset(theta_train_.cuda(), x_train_.cuda())
    score_net = NSE(theta_dim=2, x_dim=2, hidden_features=[128, 256, 128]).cuda()

    # avg_score_net = train(
    #     model=score_net,
    #     dataset=dataset,
    #     loss_fn=NSELoss(score_net),
    #     n_epochs=200,
    #     lr=1e-3,
    #     batch_size=256,
    #     prior_score=False, # learn the prior score via the classifier-free guidance approach
    # )
    # score_net = avg_score_net.module
    # torch.save(score_net, "score_net.pkl")
    score_net = torch.load("score_net.pkl")


    # # meshgrid for theta space
    theta1 = torch.linspace(-10, 10, 100)
    theta2 = torch.linspace(-10, 10, 100)

    # meshgrid 
    theta1_, theta2_ = torch.meshgrid(theta1, theta2)
    theta_ = torch.stack([theta1_.reshape(-1), theta2_.reshape(-1)], dim=-1)

    # compute learned and analytic scores
    # observation
    theta_true = torch.FloatTensor([-5, 150])  # true parameters
    x_obs = simulator(theta_true)  # x_0 ~ simulator(theta_true)
    x_obs_ = (x_obs - x_train.mean(axis=0)) / x_train.std(axis=0)

    # true posterior
    true_posterior = task.true_posterior(x_obs)
    loc = (true_posterior.loc - theta_train.mean(axis=0)) / theta_train.std(axis=0)
    cov = (
        torch.diag(1 / theta_train.std(axis=0))
        @ true_posterior.covariance_matrix
        @ torch.diag(1 / theta_train.std(axis=0))
    )
    true_posterior_ = torch.distributions.MultivariateNormal(loc=loc, covariance_matrix=cov)

    # time point
    t = torch.tensor(TIME_POINT)
    score_ana = get_vpdiff_gaussian_score(true_posterior_.loc, true_posterior_.covariance_matrix, score_net)(theta_, t=t)
    score_learned = score_net.score(theta_.cuda(), x_obs_.cuda(), t=t.cuda()).detach().cpu()

    # norm of the difference between the learned and analytic scores
    diff_score = torch.linalg.norm(score_learned - score_ana, dim=-1).reshape(100, 100)

    # plot the norm of the difference between the learned and analytic scores
    plt.contourf(theta1_, theta2_, diff_score, levels=100)
    plt.xlabel("theta_1")
    plt.ylabel("theta_2")
    plt.colorbar()
    plt.savefig(f"diff_score_t_{TIME_POINT}.png")
    plt.clf()

    print(theta_train.std(axis=0))
    print(theta_train_.std(axis=0))
    cov_train = torch.cov(theta_train_.T)
    print(cov_train)

