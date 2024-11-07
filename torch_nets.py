import numpy as np
import torch
from torch import nn


class RunningMeanStd(nn.Module):
    def __init__(self, epsilon: float = 1e-4, shape=(), *args, **kwargs):
        """
        Calulates the running mean and std of a data stream
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

        :param epsilon: helps with arithmetic issues
        :param shape: the shape of the data stream's output
        """
        super().__init__(*args, **kwargs)
        self.mean = nn.Parameter(torch.zeros(shape, dtype=torch.float), requires_grad=False)
        self.var = nn.Parameter(torch.ones(shape, dtype=torch.float), requires_grad=False)
        self.count = epsilon
        self.epsilon = nn.Parameter(torch.tensor(epsilon), requires_grad=False)

    def update(self, arr: torch.Tensor) -> None:
        batch_mean = torch.mean(arr, dim=0)
        batch_var = torch.var(arr, dim=0)
        batch_count = arr.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean: torch.Tensor, batch_var: torch.Tensor, batch_count: float) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + torch.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = m_2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean.data = new_mean
        self.var.data = new_var
        self.count = new_count

    def normalize(self, arr: torch.Tensor) -> torch.Tensor:
        return torch.clip((arr - self.mean) / torch.sqrt(self.var + self.epsilon), -5, 5)

    def unnormalize(self, arr: torch.Tensor) -> torch.Tensor:
        return arr * torch.sqrt(self.var + self.epsilon) + self.mean


class ConditionalVAE(nn.Module):
    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        instance.args = args
        instance.kwargs = kwargs
        return instance

    def __init__(self, input_size: int, hidden_size: int, latent_size: int, cond_size: int):
        super().__init__()
        self.rms_input = RunningMeanStd(shape=(input_size,))
        self.rms_cond = RunningMeanStd(shape=(cond_size,))

        self.encoder = nn.Sequential(
            nn.Linear(input_size + cond_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(cond_size + hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(cond_size + hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(cond_size + hidden_size, hidden_size),
        )
        self.decoder = MixedDecoder(cond_size, latent_size, hidden_size, input_size, num_experts=4)

        # Add mu and log_var layers for reparameterization
        self.mu = nn.Sequential(nn.LeakyReLU(), nn.Linear(hidden_size, latent_size))
        self.log_var = nn.Sequential(nn.LeakyReLU(), nn.Linear(hidden_size, latent_size))
        self.hammer_size = latent_size
        self.cond_size = cond_size
        self.input_size = input_size
        self.latent_size = latent_size

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x: torch.Tensor, w: torch.Tensor):
        x = self.rms_input.normalize(x * 1)
        w = self.rms_cond.normalize(w * 1)
        w[w.isnan()] = 0
        z = torch.cat((x, w), dim=-1)
        for j, op in enumerate(self.encoder):
            if isinstance(op, torch.nn.Linear):
                if j > 1:
                    z = torch.cat((w, z), dim=-1)
                z = op(z)
        return z

    def decode(self, z: torch.Tensor, w: torch.Tensor):
        w = self.rms_cond.normalize(w * 1)
        w[w.isnan()] = 0
        decoder_out = self.decoder(z, w)
        return decoder_out

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        encoded = self.encode(x, w)
        mu = self.mu(encoded)
        log_var = self.log_var(encoded)
        log_var = torch.clamp(log_var, -5, 5)
        z = self.reparameterize(mu, log_var)
        decoded = self.decode(z, w)
        return z, decoded, mu, log_var


class ConditionalVAEWithPrior(ConditionalVAE):
    def __init__(self, input_size: int, hidden_size: int, latent_size: int, cond_size: int):
        super().__init__(input_size, hidden_size, latent_size, cond_size)
        self.prior_encoder = nn.Sequential(
            nn.Linear(cond_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(cond_size + hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(cond_size + hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(cond_size + hidden_size, hidden_size),
        )
        self.prior_mu = nn.Sequential(nn.LeakyReLU(), nn.Linear(hidden_size, latent_size))
        self.prior_log_var = nn.Sequential(nn.LeakyReLU(), nn.Linear(hidden_size, latent_size))

    def prior_encode(self, w: torch.Tensor):
        w = self.rms_cond.normalize(w * 1)
        w[w.isnan()] = 0
        z = w * 1
        for j, op in enumerate(self.prior_encoder):
            if isinstance(op, torch.nn.Linear):
                if j > 0:
                    z = torch.cat((w, z), dim=-1)
            z = op(z)
        return z

        return encoder_out

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        # w = self.get_invariant(w)
        encoded = self.encode(x, w)
        prior_encoded = self.prior_encode(w)

        mu = self.mu(encoded)
        log_var = self.log_var(encoded)
        log_var = torch.clamp(log_var, -5, 5)

        prior_mu = self.prior_mu(prior_encoded)
        prior_log_var = self.prior_log_var(prior_encoded)
        prior_log_var = torch.clamp(prior_log_var, -5, 5)

        z = self.reparameterize(mu, log_var)
        decoded = self.decode(z, w)
        return z, decoded, mu, log_var, prior_mu, prior_log_var


class MixedDecoder(nn.Module):
    def __init__(
        self,
        cond_size,
        latent_size,
        hidden_size,
        output_size,
        num_experts,
    ):
        super().__init__()

        input_size = latent_size + cond_size
        inter_size = latent_size + cond_size + hidden_size
        self.decoder_layers = [
            (
                nn.Parameter(torch.empty(num_experts, input_size, hidden_size)),
                nn.Parameter(torch.empty(num_experts, hidden_size)),
                nn.LeakyReLU(),
            ),
            (
                nn.Parameter(torch.empty(num_experts, inter_size, hidden_size)),
                nn.Parameter(torch.empty(num_experts, hidden_size)),
                nn.LeakyReLU(),
            ),
            (
                nn.Parameter(torch.empty(num_experts, inter_size, hidden_size)),
                nn.Parameter(torch.empty(num_experts, hidden_size)),
                nn.LeakyReLU(),
            ),
            (
                nn.Parameter(torch.empty(num_experts, inter_size, hidden_size)),
                nn.Parameter(torch.empty(num_experts, hidden_size)),
                nn.LeakyReLU(),
            ),
            (
                nn.Parameter(torch.empty(num_experts, inter_size, output_size)),
                nn.Parameter(torch.empty(num_experts, output_size)),
                None,
            ),
        ]

        for index, (weight, bias, _) in enumerate(self.decoder_layers):
            index = str(index)
            torch.nn.init.kaiming_uniform_(weight)
            bias.data.fill_(0.01)
            self.register_parameter("w" + index, weight)
            self.register_parameter("b" + index, bias)

        # Gating network
        gate_hsize = 64
        self.gate = nn.Sequential(
            nn.Linear(input_size, gate_hsize),
            nn.LeakyReLU(),
            nn.Linear(gate_hsize, gate_hsize),
            nn.LeakyReLU(),
            nn.Linear(gate_hsize, num_experts),
        )

    def forward(self, z, c):
        coefficients = torch.softmax(self.gate(torch.cat((z, c), dim=-1)), dim=-1)
        layer_out = c * 1

        for i, (weight, bias, activation) in enumerate(self.decoder_layers):
            flat_weight = weight.flatten(start_dim=1, end_dim=2)
            mixed_weight = torch.matmul(coefficients, flat_weight).view(*coefficients.shape[:-1], *weight.shape[1:3])

            if i == 0:
                input = torch.cat((z, layer_out), dim=-1)
            else:
                input = torch.cat((z, c, layer_out), dim=-1)
            mixed_bias = torch.matmul(coefficients, bias)
            out = torch.baddbmm(mixed_bias, input, mixed_weight[:, 0])
            layer_out = activation(out) if activation is not None else out

        return layer_out


class TransformerDenoiser(nn.Module):
    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        instance.args = args
        instance.kwargs = kwargs
        return instance

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_heads: int,
        num_layers: int,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = 0.0
        self.activation = nn.GELU()
        self.temb = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.deproj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, input_size),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_size,
            dropout=self.dropout,
            activation=self.activation,
            batch_first=True,
            bias=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, self.num_layers)
        self.rms_input = RunningMeanStd(shape=(input_size,))

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.proj(self.rms_input.normalize(x))
        t = self.temb(t)
        xt = torch.cat([x, t], dim=1)

        pe = torch.zeros_like(xt)
        idxs = torch.arange(xt.shape[1], device=xt.device)
        div_term = torch.exp(torch.arange(0, xt.shape[-1] // 2, 1).float() * (-np.log(1000.0) / (xt.shape[-1]))).to(xt.device)
        pe[..., 0::2] = torch.sin(idxs[..., None] * div_term[None])
        pe[..., 1::2] = torch.cos(idxs[..., None] * div_term[None])
        xt += pe

        xt = self.encoder(xt)
        xt = self.deproj(xt)
        return xt[:, :-1]


class UnconditionalEDM(nn.Module):

    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        instance.args = args
        instance.kwargs = kwargs
        return instance

    def __init__(
        self,
        input_size: int,
        input_frames: int,
        hidden_size: int,
        latent_size: int,
        num_heads: int,
        num_layers: int,
    ):
        super().__init__()

        self.input_size = input_size
        self.input_frames = input_frames
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.time_size = 1

        self.denoiser = TransformerDenoiser(self.input_size, self.hidden_size, self.num_heads, self.num_layers)
        self.rms_input = RunningMeanStd(shape=(input_size,))
        self.mse_loss = nn.MSELoss()
        self.sigma_min = 0.002
        self.sigma_max = 80

    def forward(self, x: torch.Tensor):
        """
        Use forward diffusion to corrupt the clean signal `x`,
        and train the model to predict the corrupted signal.
        Eps model is aware of the amount of corruption involved,
        i.e. number of forward diffusion steps used and thus how much noise level there must be.
        """

        eps = torch.randn_like(x)

        # EDM: sigma approach
        log_sigma = -1.2 + torch.randn((x.shape[0],), device=x.device) * 1.2
        log_sigma = log_sigma.view(-1, *(1 for _ in x.shape[1:]))
        sigma = torch.exp(log_sigma)
        sigma = torch.clamp(sigma, self.sigma_min, self.sigma_max)
        sigma_data = 0.5
        c_skip = sigma_data**2 / (sigma**2 + sigma_data**2)
        c_out = sigma * sigma_data / torch.sqrt(sigma_data**2 + sigma**2)
        c_in = 1 / torch.sqrt(sigma**2 + sigma_data**2)
        c_noise = 1 / 4 * torch.log(sigma)
        lammy = (sigma**2 + sigma_data**2) / (sigma * sigma_data) ** 2

        noised_x = x + eps * sigma
        eps_tar = x
        eps_out = c_skip * noised_x + c_out * self.denoise(c_in * noised_x, c_noise).view(x.shape)
        return eps_out, eps_tar, lammy.view(-1)

    def denoise(self, x: torch.Tensor, t: torch.Tensor):
        return self.denoiser(x, t)

    def step(self, x_t: torch.Tensor, tt: int):
        device = x_t.device
        t = torch.ones((x_t.shape[0], *(1 for _ in range(len(x_t.shape) - 2)), 1), device=device) * tt
        sigma = torch.clamp(t, self.sigma_min, self.sigma_max)
        sigma_data = 0.5
        c_skip = sigma_data**2 / (sigma**2 + sigma_data**2)
        c_out = sigma * sigma_data / torch.sqrt(sigma_data**2 + sigma**2)
        c_in = 1 / torch.sqrt(sigma**2 + sigma_data**2)
        c_noise = 1 / 4 * torch.log(sigma)

        eps_out = c_skip * x_t + c_out * self.denoise(c_in * x_t, c_noise).view(x_t.shape)

        return eps_out
