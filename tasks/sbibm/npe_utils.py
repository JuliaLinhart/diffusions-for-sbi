from sbi.utils import match_theta_and_x_batch_shapes


def fwd_flow_transform_obs(batch_z, observation, flow):
    observation_emb = flow._embedding_net(observation)
    z_repeated, x_repeated = match_theta_and_x_batch_shapes(batch_z, observation_emb)
    z_transformed = flow._transform.inverse(z_repeated, x_repeated)[0].detach()
    return z_transformed


def inv_flow_transform_obs(batch_theta, observation, flow):
    observation_emb = flow._embedding_net(observation)
    theta_repeated, x_repeated = match_theta_and_x_batch_shapes(
        batch_theta, observation_emb
    )
    theta_transformed = flow._transform(theta_repeated, x_repeated)[0].detach()
    return theta_transformed


def sample_from_npe_obs(
    npe,
    observation,
    n_samples=None,
    base_dist_samples=None,
):
    if base_dist_samples is None and n_samples is None:
        raise ValueError("Either base_dist_samples or n_samples must be given")
    else:
        if base_dist_samples is None:
            base_dist_samples = npe.posterior_estimator._distribution.sample(n_samples)
        if n_samples is None:
            n_samples = base_dist_samples.shape[0]

    assert base_dist_samples.shape[0] == n_samples

    # Set default x_0 for npe
    npe.set_default_x(observation)
    # Sample from npe at x_0
    npe_samples_obs = fwd_flow_transform_obs(
        batch_z=base_dist_samples, observation=observation, flow=npe.posterior_estimator
    )
    return npe_samples_obs
