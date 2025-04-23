# Denoising Diffusion Probabilistic Models (DDPM)

## Overview

This project explores the connection between Denoising Score Matching (DSM) and Denoising Diffusion Probabilistic Models (DDPM). The report delves into how Langevin Dynamics can generate samples from an estimated score, the challenges posed by inaccessible true densities, and how DSM addresses these issues by adding noise to data. It further explains Noise Conditional Score Networks (NCSNs) and shows that DDPMs can be viewed as a special case of DSM, providing a unified perspective on score-based generative modeling.

### Langevin Dynamics

Langevin Dynamics is used to generate new samples from a dataset using the score function $\nabla_{x} \log (p(\mathbf{x}))$. The dynamics are defined as:

$$
\tilde{\mathbf{x}}_{t}=\tilde{\mathbf{x}}_{t-1}+\frac{\varepsilon}{2} \nabla_{\mathbf{x}} \log \left(p\left(\tilde{\mathbf{x}}_{t-1}\right)\right)+\sqrt{\varepsilon} \mathbf{z}_{t}
$$

where $\mathbf{z}_{t} \sim \mathcal{N}(0, \mathbf{I})$ and $\varepsilon > 0$ is a fixed step size.

### Denoising Score Matching (DSM)

DSM estimates the score function by adding noise to the data, making the estimation feasible throughout the input space. The objective is:

$$
\min _{\theta} \frac{1}{2} \mathbb{E}_{q_{\sigma}(\tilde{\mathbf{x}} \mid \mathbf{x}) p_{\text {data }}(\mathbf{x})}\left(\left\|s_{\theta}(\tilde{\mathbf{x}})-\nabla_{\tilde{x}} \log q_{\sigma}(\tilde{\mathbf{x}} \mid \mathbf{x})\right\|_{2}^{2}\right)
$$

### Noise Conditional Score Networks (NCSNs)

NCSNs train a model on multiple noise levels, enabling annealed Langevin sampling that gradually removes noise. The training objective is:

$$
\mathcal{L}\left(\theta ;\left\{\sigma_{i}\right\}_{i=1}^{L}\right)=\frac{1}{L} \sum_{i=1}^{L} \lambda\left(\sigma_{i}\right) \ell\left(\theta ; \sigma_{i}\right)
$$

### Denoising Diffusion Probabilistic Models (DDPM)

DDPMs reverse a forward diffusion process, which can be seen as a special case of DSM. The forward process is defined as:

$$
q\left(\mathbf{x}_{t} \mid \mathbf{x}_{t-1}\right)=\mathcal{N}\left(\mathbf{x}_{t} ; \sqrt{1-\beta_{t}} \mathbf{x}_{t-1}, \beta_{t} \mathbf{I}\right)
$$

The backward process involves learning to recover the original data sample from the noised version.