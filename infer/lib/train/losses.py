import torch


def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            rl = rl.float().detach()
            gl = gl.float()
            loss += torch.mean(torch.abs(rl - gl))

    return loss * 2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        dr = dr.float()
        dg = dg.float()
        r_loss = torch.mean((1 - dr) ** 2)
        g_loss = torch.mean(dg**2)
        loss += r_loss + g_loss
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        dg = dg.float()
        l = torch.mean((1 - dg) ** 2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses


def kl_loss(z_p, logs_q, m_p, logs_p, z_mask):
    """
    z_p, logs_q: [b, h, t_t]
    m_p, logs_p: [b, h, t_t]
    """
    z_p = z_p.float()
    logs_q = logs_q.float()
    m_p = m_p.float()
    logs_p = logs_p.float()
    z_mask = z_mask.float()

    kl = logs_p - logs_q - 0.5
    kl += 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2.0 * logs_p)
    kl = torch.sum(kl * z_mask)
    l = kl / torch.sum(z_mask)
    return l


def kl_loss_gaussian(m_p, logvars_p, m_q, logvars_q, z_mask):
    """
    Computes the KL divergence between two multivariate normal distributions with diagonal covariance.

    Parameters:
    - m_p: Tensor of shape (..., D), mean of distribution p
    - logvars_p: Tensor of shape (..., D), log variance of distribution p
    - m_q: Tensor of shape (..., D), mean of distribution q
    - logvars_q: Tensor of shape (..., D), log variance of distribution q

    Returns:
    - kl_loss: Tensor of shape (...), KL divergence for each sample
    """
    m_q = m_q.float()
    logvars_q = logvars_q.float()
    m_p = m_p.float()
    logvars_p = logvars_p.float()
    z_mask = z_mask.float()

    # Small constant for numerical stability
    eps = 1e-6

    # Clamp log variances to prevent numerical issues in exp and log operations
    logvars_p = torch.clamp(logvars_p, min=-30.0, max=20.0)
    logvars_q = torch.clamp(logvars_q, min=-30.0, max=20.0)

    # Compute variances (sigma squared)
    sigma_p2 = torch.exp(logvars_p)
    sigma_q2 = torch.exp(logvars_q)

    # Ensure variances are not zero by adding epsilon
    sigma_p2 = sigma_p2 + eps
    sigma_q2 = sigma_q2 + eps

    # Compute the log variance ratio
    log_var_ratio = logvars_q - logvars_p

    # Compute squared difference of means
    mean_diff_squared = (m_p - m_q) ** 2

    # Compute KL divergence elements
    kl_elements = log_var_ratio + (sigma_p2 + mean_diff_squared) / sigma_q2 - 1

    kl = 0.5 * kl_elements

    kl = torch.sum(kl * z_mask)
    l = kl / torch.sum(z_mask)
    return l
