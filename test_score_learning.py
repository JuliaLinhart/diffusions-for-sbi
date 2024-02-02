import torch
from nse import NSE
from sm_utils import train
import tqdm
from functools import partial
from tasks.toy_examples.data_generators import Gaussian_Gaussian_mD
from score_model_vp import ScoreModel, FNet
import collections
import matplotlib.pyplot as plt
from matplotlib import cm


def build_R2_cross_valid(task, prior_, x_val_, theta_val_, score_model, n_samples=32):
    t = torch.linspace(0, 1, 500)
    lik_cov = task.simulator_cov
    lik_cov_ = (
            torch.diag(1 / theta_train.std(axis=0))
            @ lik_cov
            @ torch.diag(1 / theta_train.std(axis=0))
    )
    posterior_cov_0_ = torch.linalg.inv(
        (torch.linalg.inv(lik_cov_) + torch.linalg.inv(prior_.covariance_matrix)))

    posterior_mean_0_ = (posterior_cov_0_ @ (torch.linalg.inv(prior_.covariance_matrix) @ prior_.loc[:, None].repeat(1, x_val_.shape[0]) +
                                             torch.linalg.inv(lik_cov_) @ x_val_.mT)).mT

    posterior_cov_diffused_ = ((posterior_cov_0_[None] * score_model.alpha(t)[:, None, None]+
                                (1 - score_model.alpha(t))[:, None, None] * torch.eye(posterior_cov_0_.shape[0])[
                                    None])).cpu()
    posterior_mean_diffused_ = (posterior_mean_0_[None] * (score_model.alpha(t)**.5)[:, None, None]).cpu()
    dist = torch.distributions.MultivariateNormal(loc=posterior_mean_diffused_[1:],
                                                  covariance_matrix=posterior_cov_diffused_[1:, None])
    def calculate_mse_scores(score_model):
        ref_samples = dist.sample((n_samples,))
        ref_samples.requires_grad_(True)
        dist.log_prob(ref_samples).sum().backward()
        real_score = ref_samples.grad.clone()
        ref_samples.grad = None
        ref_samples.requires_grad_(False)
        approx_score_gauss = score_model.score(ref_samples.reshape(-1, ref_samples.shape[-1]).cuda(),
                                               x_val_.cuda().unsqueeze(0).unsqueeze(1).repeat(n_samples, 499, 1, 1).reshape(-1, ref_samples.shape[-1]),
                                               t[None, 1:, None, None].repeat(n_samples, 1, x_val_.shape[0], 1).reshape(-1, 1).cuda())
        approx_score_gauss = approx_score_gauss.reshape(*ref_samples.shape).cpu()
        error_mean = torch.linalg.norm(real_score - real_score.mean(dim=0)[None], dim=-1)**2
        errors = (approx_score_gauss - real_score).square().sum(dim=-1)
        r2_score_gauss = (1 - (errors.sum(dim=0) / error_mean.sum(dim=0))).mean(dim=-1)
        return r2_score_gauss.detach()
    return calculate_mse_scores


def build_kl_cross_valid(task, prior_, x_val_, theta_val_, score_model, n_samples=32):
    def calculate_kl_scores(score_model):
        with torch.no_grad():
            return "kl", score_model.loss(theta_val_.repeat(10, 1).cuda(), x_val_.repeat(10, 1).cuda()).cpu().item()
    return calculate_kl_scores



if __name__ == "__main__":
    #from tasks.toy_examples.data_generators import SBIGaussian2d
    #from nse import NSE, NSELoss
    from sm_utils import train



    torch.manual_seed(1)
    N_TRAIN = 10_000
    N_VAL = 10
    N_SAMPLES = 4096
    BATCH_SIZE = 2048
    r2_profiles = {}
    losses = {}
    colormap = lambda d: cm.get_cmap('rainbow')(d / 64)
    for DIM in [2, 8, 16, 32, 64]:
        # Observations
        torch.manual_seed(42)
        means = torch.rand(DIM) * 20 - 10  # between -10 and 10
        stds = torch.rand(DIM) * 25 + 0.1  # between 0.1 and 25.1
        task = Gaussian_Gaussian_mD(dim=DIM, means=means, stds=stds)
        prior = task.prior
        simulator = task.simulator
        theta_true = prior.sample(sample_shape=(1,))  # true parameters

        x_obs = simulator(theta_true)  # x_obs ~ simulator(theta_true)
        x_obs_100 = torch.cat(
            [simulator(theta_true).reshape(1, -1) for _ in range(100)], dim=0
        )

        # True posterior: p(theta|x_obs)
        true_posterior = task.true_posterior(x_obs)

        # Train data
        theta_train = task.prior.sample((N_TRAIN,))
        x_train = simulator(theta_train)

        # Validation
        theta_val = task.prior.sample((N_VAL,))
        x_val = simulator(theta_val)

        # normalize theta
        theta_train_ = (theta_train - theta_train.mean(axis=0)) / theta_train.std(axis=0)
        theta_val_ = (theta_val - theta_train.mean(axis=0)) / theta_train.std(axis=0)


        # normalize x
        x_train_ = (x_train - x_train.mean(axis=0)) / x_train.std(axis=0)
        x_obs_ = (x_obs - x_train.mean(axis=0)) / x_train.std(axis=0)
        x_obs_100_ = (x_obs_100 - x_train.mean(axis=0)) / x_train.std(axis=0)
        x_val_ = (x_val - x_train.mean(axis=0)) / x_train.std(axis=0)

        # # train score network
        dataset = torch.utils.data.TensorDataset(theta_train_.cuda(), x_train_.cuda())
        score_model = NSE(x_dim=DIM, theta_dim=DIM, sampling_dist='Uniform').cuda()
        # normalize prior
        loc_ = (prior.loc - theta_train.mean(axis=0)) / theta_train.std(axis=0)
        cov_ = (
                torch.diag(1 / theta_train.std(axis=0))
                @ prior.covariance_matrix
                @ torch.diag(1 / theta_train.std(axis=0))
        )
        prior_ = torch.distributions.MultivariateNormal(
            loc=loc_, covariance_matrix=cov_
        )
        # Normalize posterior
        loc_ = (true_posterior.loc - theta_train.mean(axis=0)) / theta_train.std(axis=0)
        cov_ = (
                torch.diag(1 / theta_train.std(axis=0))
                @ true_posterior.covariance_matrix
                @ torch.diag(1 / theta_train.std(axis=0))
        )
        # true_posterior_ = torch.distributions.MultivariateNormal(
        #     loc=loc_.cuda(), covariance_matrix=cov_.cuda()
        # )
        r2_score_loss_fn = build_R2_cross_valid(task=task,
                                                prior_=prior_,
                                                x_val_=x_val_,
                                                theta_val_=theta_val_,
                                                score_model=score_model.cpu())
        kl_score_loss_fn = build_kl_cross_valid(task=task,
                                                prior_=prior_,
                                                x_val_=x_val_,
                                                theta_val_=theta_val_,
                                                score_model=score_model.cpu())
        score_model.net, losses[DIM] = train(nse=score_model.cuda(),
                                             dataset=dataset,
                                             n_epochs=300,
                                             lr=1e-4,
                                             batch_size=64,
                                             prior_score=False,
                                             val_fn=kl_score_loss_fn,
                                             val_period=10)
        score_model.net.eval()
        # samples_ = score_model.ddim((100,),
        #                             x=x_val_[0:1].repeat(100, 1).cuda(),
        #                             eta=.1,
        #                             n_steps=1000
        #                             )
        samples_ = score_model.euler(shape=(100,),
                                     x=x_val_[0:1].repeat(100, 1).cuda(),
                                     steps=1000
                                     ).detach()
        samples = samples_.cpu() * theta_train.std(dim=0)[None] + theta_train.mean(dim=0)[None]
        true_posterior = task.true_posterior(x_val[0:1])
        ref_samples = true_posterior.sample((100,))[:, 0, :2]
        plt.scatter(*samples[:, :2].cpu().T, label='Samples')
        plt.scatter(*ref_samples.cpu().T, label='Ref')
        plt.xlim(ref_samples[:, 0].min(), ref_samples[:, 0].max())
        plt.ylim(ref_samples[:, 1].min(), ref_samples[:, 1].max())
        plt.legend()
        plt.show()
        r2_profiles[DIM] = r2_score_loss_fn(score_model=score_model).cpu()
        fig, axes = plt.subplots(2, 1)
        for d, l in losses.items():
            c = colormap(d)
            n_iter_train = list(range(len(l["train"])))
            n_iter_val = list(range(0, len(l["train"]), len(l["train"])//len(l["val"])))
            axes[0].plot(n_iter_train, l["train"], color=c, linestyle='solid')
            axes[0].plot(n_iter_val, l["val"], color=c, linestyle='dashed')
        axes[0].set_yscale('log')
        axes[0].set_ylabel('KL')
        for d, r2 in r2_profiles.items():
            c = colormap(d)
            axes[1].plot(r2, color=c, linestyle='solid', label=d)
        axes[1].legend()
        axes[1].set_ylim(-1, 1)
        axes[1].set_ylabel('R2 score')
        axes[1].set_xlabel('Diffusion step')
        fig.show()

        # model_name = f"score_net_gauss_{DIM}.pkl"
        # score_model.save(model_name, {"loss": losses[DIM],
        #                               "r2": r2_profiles[DIM]})

