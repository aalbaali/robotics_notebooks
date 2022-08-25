### A Pluto.jl notebook ###
# v0.19.5

using Markdown
using InteractiveUtils

# ╔═╡ f83a18ee-ede4-11ec-01a5-894d39d2f38a
begin
	using FFTW
	using Plots
	using Statistics
	using StatsBase;
	using DSP;
end

# ╔═╡ ac471330-a90c-459d-8832-d4b94e5fb31a
# Signal length
N = 1000;

# ╔═╡ 5f308103-d472-40b4-b3f1-c642a006ba14
# Number of trials (i.e., number of signal realizations)
N_trials = 100;

# ╔═╡ c5db2eab-d903-4977-a9db-ed9599755bbf
σ = 0.25;

# ╔═╡ b588624a-528a-4d89-8c74-e372e1b05f93
# White noise
signals = [10 .+ σ * randn(N) for i in 1:N_trials];

# ╔═╡ 128d15d4-c59e-492d-b1aa-7a54b51f90c6
# Function to the i-th element of every signal
get_signals_idx(i::Int, signals) = [signal[i] for signal in signals];

# ╔═╡ e4c08a74-0f32-4680-b2ab-a520b795c11d
get_signals_idx(i::Int) = get_signals_idx(i, signals);

# ╔═╡ 41ae5667-e0aa-43b3-a540-3fe2ecf43e02
signals_idx = get_signals_idx.(1:N)

# ╔═╡ d58057d1-b8de-4a93-81c5-cc5600498058
stds = std.(signals_idx);

# ╔═╡ e5a835e6-2927-4769-b8eb-0389b0f2ff98
plot(stds)

# ╔═╡ 95f5bd61-0414-4699-a2fa-0c0a9270beef
# Imperical autocovariance function
begin
	acov(i₁::Int, i₂::Int) = cov(signals_idx[i₁], signals_idx[i₂]);
	acor(i₁::Int, i₂::Int) = cor(signals_idx[i₁], signals_idx[i₂]);
end

# ╔═╡ aaa72077-8a41-46dd-a04c-9aa90925710f
begin
	acov(τ::Int) = acov(1, τ);
	acor(τ::Int) = acor(1, τ);
end

# ╔═╡ 40ac99ac-fc51-42f3-9143-db5917192ed1
begin
	plot(acov.(1:N))
	# plot!([[acov(i1, i) for i in 1:N] for i1 in 1:100:1000])
end

# ╔═╡ 7dede67b-2593-478c-8384-476f7be90b3e
std(signals[1])

# ╔═╡ 2857a892-9770-4e52-b10c-82502d7f54c0
acov.(1:N)

# ╔═╡ cb2cc351-d0d4-497e-b0db-1f6f80d10a73
sqrt(mean(abs.(fft(acor.(1:N)))) / N)

# ╔═╡ a00a243b-772b-4ff9-8f73-4af1f32dd600
σ^2

# ╔═╡ b4c3cb24-9091-4bdf-a35a-54cda55198df
acor_val = autocor(signals[1])

# ╔═╡ 1528838e-13a6-4c04-9ba5-d1e496e77cf7
acov_val = autocov(signals[1])

# ╔═╡ 79842cf2-71ec-4531-9841-2c5a122c8485
acov_val

# ╔═╡ aca2ef31-05eb-46cc-bff2-2a1ccbca1347
acor_tmp(τ::Int) = cor(signal_cut[1], signal_cut[τ]);

# ╔═╡ 50495c29-d138-4747-ba9a-287db321be1d
begin
	a = zeros(N);
	a[1] = 3^2;
end

# ╔═╡ ac1b0137-c6af-4bd6-9712-349bbd145f3c
A = fft(a);

# ╔═╡ f5834193-04e9-4f70-91d2-74586306b419
ifft(3.0 * ones(1000))

# ╔═╡ 1f5c6c36-1f0e-40b4-86a8-f5023446dfea
mean(abs.(fft(autocov(3.0 * randn(100000)))))

# ╔═╡ 0a5284f1-14fa-4275-ba1f-8403b62299b0
mean(abs.(fft(autocov(s))))

# ╔═╡ d0945d2d-aa1e-4de4-8252-2f401ce62f03
autocov(s)

# ╔═╡ 58965450-ed46-405b-bf03-1a7db8e0b786
# From https://discourse.julialang.org/t/autocov-with-fft/4178/7
function autocov_con(x::AbstractVector{<:Real},lags::UnitRange{Int})
    lx = size(x,1)
    x .-= mean(x)
    A = conv(x,reverse(x))/lx
    A = [A[k + lx] for k in lags]
end

# ╔═╡ 4f0e4f9e-4ec9-414b-b118-d0bb040fc2cc
function autocor_con(x::AbstractVector{<:Real},lags::UnitRange{Int})
    lx = length(x)    
    A = conv(x,reverse(x))/lx
    A = [A[k + lx] for k in lags]
end

# ╔═╡ 577846ba-ab0d-40ff-858e-5945b1229c9d
begin
	lags = 0:convert(Int, min(size(s,1)-1, 10*log10(size(s,1))))
	autocor_con(s, lags)
end

# ╔═╡ 2accedc7-4ea4-4eff-838d-0b24a196b489
mean(abs.(fft(autocor_con(s, 0:999))))

# ╔═╡ d4354bc0-8b7c-41e2-bba8-7c40468ebaf9
autocov(s)

# ╔═╡ 9977af4e-d2b5-4402-b7f7-d05415d6fa47
md"""
# Summary
- The power spectral density (PSD) of a wide sense stationary signal (WSS) $w_{n}$ is the Fourier transform of its autocorrelation function
```math
\operatorname{cov}(w_{n_{1}}, w_{n_{2}}) = R_{w}(n_{2} - n_{1}) = \sigma_{w}^{2}\delta_{k}.
```
- The autocorrelation of a discrete-time signal is [given by](https://www.egr.msu.edu/classes/ece458/radha/ss07Keyur/Lab-Handouts/PSDESDetc.pdf)
```math
R_{w}[\tau] = \sum_{n} w[n]w[n-\tau] = w[n] * w[-n].
```
Check the function `autocor_con` to see how the autocorrelation is computed, which is similar to how the autocovariance is computed with the exception of subtracting the mean.

- The PSD can then be computed using the Fourier transform
```math
S_{w}(\omega) = \mathcal{F}\left\{R_{w}[\tau]\right\} = \sigma_{w}^2, \quad \forall \omega\in[-\pi, \pi),
```
which can be approximated using
```julia
mean(abs.(fft(autocov_con(w))))
```
Note that a different frequency unit may be required, which would replace `mean` with some "integration" function.
For example, it can be as simple as `sum(...) * f_s`, where `f_s` is the sampling frequency, in appropriate units.
"""

# ╔═╡ c097e9b0-8bd6-44ad-ab76-5bfb7db64a11
md"""
# References
- Farrel
- Oppenheimer
- https://discourse.julialang.org/t/autocov-with-fft/4178/7
- https://juliastats.org/StatsBase.jl/stable/signalcorr/#StatsBase.autocor
- https://www.egr.msu.edu/classes/ece458/radha/ss07Keyur/Lab-Handouts/PSDESDetc.pdf
"""

# ╔═╡ Cell order:
# ╠═f83a18ee-ede4-11ec-01a5-894d39d2f38a
# ╠═ac471330-a90c-459d-8832-d4b94e5fb31a
# ╠═5f308103-d472-40b4-b3f1-c642a006ba14
# ╠═c5db2eab-d903-4977-a9db-ed9599755bbf
# ╠═b588624a-528a-4d89-8c74-e372e1b05f93
# ╠═128d15d4-c59e-492d-b1aa-7a54b51f90c6
# ╠═e4c08a74-0f32-4680-b2ab-a520b795c11d
# ╠═41ae5667-e0aa-43b3-a540-3fe2ecf43e02
# ╠═d58057d1-b8de-4a93-81c5-cc5600498058
# ╠═e5a835e6-2927-4769-b8eb-0389b0f2ff98
# ╠═95f5bd61-0414-4699-a2fa-0c0a9270beef
# ╠═aaa72077-8a41-46dd-a04c-9aa90925710f
# ╠═40ac99ac-fc51-42f3-9143-db5917192ed1
# ╠═7dede67b-2593-478c-8384-476f7be90b3e
# ╠═2857a892-9770-4e52-b10c-82502d7f54c0
# ╠═cb2cc351-d0d4-497e-b0db-1f6f80d10a73
# ╠═a00a243b-772b-4ff9-8f73-4af1f32dd600
# ╠═b4c3cb24-9091-4bdf-a35a-54cda55198df
# ╠═1528838e-13a6-4c04-9ba5-d1e496e77cf7
# ╠═79842cf2-71ec-4531-9841-2c5a122c8485
# ╠═aca2ef31-05eb-46cc-bff2-2a1ccbca1347
# ╠═50495c29-d138-4747-ba9a-287db321be1d
# ╠═ac1b0137-c6af-4bd6-9712-349bbd145f3c
# ╠═f5834193-04e9-4f70-91d2-74586306b419
# ╠═1f5c6c36-1f0e-40b4-86a8-f5023446dfea
# ╠═0a5284f1-14fa-4275-ba1f-8403b62299b0
# ╠═d0945d2d-aa1e-4de4-8252-2f401ce62f03
# ╠═58965450-ed46-405b-bf03-1a7db8e0b786
# ╠═4f0e4f9e-4ec9-414b-b118-d0bb040fc2cc
# ╠═577846ba-ab0d-40ff-858e-5945b1229c9d
# ╠═2accedc7-4ea4-4eff-838d-0b24a196b489
# ╠═d4354bc0-8b7c-41e2-bba8-7c40468ebaf9
# ╟─9977af4e-d2b5-4402-b7f7-d05415d6fa47
# ╟─c097e9b0-8bd6-44ad-ab76-5bfb7db64a11
