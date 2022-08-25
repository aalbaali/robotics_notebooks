### A Pluto.jl notebook ###
# v0.19.5

using Markdown
using InteractiveUtils

# ╔═╡ d9160695-c11a-4e69-94fc-26e4302e08e4
begin
	using PlutoUI
	PlutoUI.TableOfContents(title="📚 Table of Contents", indent=true, depth=4, aside=true)
end

# ╔═╡ f31d07b1-d65d-47f6-b23c-fd7c0b58a506
begin
	using LinearAlgebra
	using Plots
	using Statistics
	using SignalAnalysis
	using FFTW
end

# ╔═╡ 831f8744-d659-11ec-3d14-090c4a016610
md"""
# Overview
Recently, I've been trying implement navigation algorithms on real robots using real sensors data, which requires looking into things that may be trivial when running a simulation but not so trivial when running on real data.
For example, what is the covariance of the signal noise?
Wait... is there a *covariance* on a *continuous* random signal/process?
Is that what the notion of *autocovariance* comes in?
It's possibly related to the power spectral density (PSD).
So, what is a PSD?

There are too many question to answer, but they stem around random processes and how to characterize them.
In other words, I'm trying to understand the generlization of random *variables* into random *processes*.

## Goals of this notebook
- Understand Fourier transforms what they represent. Specifically, [discrete Fourier transforms (DFT)](https://en.wikipedia.org/wiki/Discrete_Fourier_transform) and [fast Fourier transforms (FFT)](https://en.wikipedia.org/wiki/Fast_Fourier_transform).
- Understand what are PSDs and how to compute them.
- Use PSD to get discrete-time covariances.
"""

# ╔═╡ 5dec98a0-95b1-4cf5-a71f-27f3d4ac8139
md"""
# Fourier Transform (FT)
The Fourier transform takes a continuous signal $x(t)\in\mathbb{R}$ as a function of time $t\in\mathbb{R}$ and decomposes it into its *frequency* components $X(\xi)\in\mathbb{C}$, denoted by an upper case in this document.
The frequencies could be [temporal](https://en.wikipedia.org/wiki/Frequency) (i.e., time-based), as in they could have units of cycles/sec, which are also referred to as Hz, or they could be [spatial](https://en.wikipedia.org/wiki/Spatial_frequency). For example, they could have units of cycles/meter.

The [Fourier transform](https://en.wikipedia.org/wiki/Fourier_transform) is defined as
```math
\begin{align}
X(\xi)
:=
\mathcal{F}(x(t))
=
\int_{-\infty}^{\infty}
x(t)e^{-2\pi\xi t \jmath}\mathrm{d}t
\end{align} \in \mathbb{C},
```
where the Fourier transform at each frequency $\xi$ is a complex number $X(\xi)\in\mathbb{C}$.

To retrieve the signal $x(t)$ from the Fourier transform $X(\xi)$, the inverse Fourier transform is used, which is given by
```math
\begin{align}
x(t)
=
\int_{-\infty}^{\infty}X(\xi)e^{2\pi\omega t\xi}
\mathrm{d}\xi.
\end{align}
```
"""

# ╔═╡ f4a89cf7-6c0e-43b3-b09b-81669d9c280d
md"""

## Why complex numbers?
The *frequency components* refer to the components that characterize a sinusoidal function with a given frequency.
More specifically, a sinusoidal function
```math
\begin{align}
y(t; A, \xi, \theta) = A\cos(2\pi\xi t + \theta)
\end{align}
```
is characterized by its frequency $\xi\in\mathbb{R}_{+}$, amplitude $A\in\mathbb{R}_{+}$, and phase $\theta\in[0,2\pi)$.
Therefore, for a given frequency $\xi$, the sinusoidal function is described by its amplitude $A\in\mathbb{R}_{+}$ and phase $\theta\in[0,2\pi)$.
The two variables can be described nicely using a complex number $z\in\mathbb{C}$, where the magnitude represents the amplitude and the argument represents the phase.
That is,
```math
A = \vert z\vert \in \mathbb{R}_{+},
\qquad
\theta = \tan^{-1}\left(\frac{\operatorname{Re}(z)}{\operatorname{Im}(z)}\right).
```

Dealing with a single complex variable is perhaps easier than working with two (real) variables, one of which belongs to a special domain $[0, 2\pi)$ (i.e., the phase).
Therefore, the Fourier transform uses complex numbers to characterize sinusoidal functions.

The output of Fourier transform $X(\jmath\omega)$ is a complex number that represents the *amplitude* and *phase* of a sinusoidal function with frequency $\xi$.
The *amplitude* is interpreted as the contribution of such sinusoidal function into the temporal signal $x(t)$.

For example, a signal
```math
x(t) = \cos(2\pi t) + 3\sin(2\pi 2 t) = \cos(2\pi t) + 3\cos(2\pi 2 t + \pi/2),
```
will have the Fourier transform
```math
X(\xi) =
\begin{cases}
1 + \jmath 0, & \xi = 1, \\
0 - \jmath 3 , & \xi = 2, \\
0, & \text{otherwise}.
\end{cases}
```

NEEDS CLARIFICATION HOW THESE NUMBERS WERE COMPUTED!
"""

# ╔═╡ 24a64d23-2a86-41a9-9350-760838bcb186
md"""
# Discrete-time Fourier transform (DTFT)
In practice, signals a recorded as *discrete* signals, not continuous.
Discretized signals are usually referred to as *sequences*.
The [discrete-time Fourer transform](https://en.wikipedia.org/wiki/Discrete-time_Fourier_transform) (DTFT) is introduced as a Fourier transform on sequences.
The *discrete-time* Fourier transform takes a discrete-time signal $x[n]\in\mathbb{R}$ where $n\in\mathbb{Z}$, and returns a *continuous-time* transform $X(\xi)\in\mathbb{C}$ (i.e., $\xi\in\mathbb{R}_{+}$).
This is different from the discrete Fourier transform discussed in the next section, which takes the same type of input but instead returns a *discrete-time* transform $X[k]\in\mathbb{C}$, where $k\in\mathbb{Z}$.

A discrete-time sequence $x[n]$ can be sampled from a continuous-time signal $x_{c}(t)$ through an *ideal continuous-to-discrete-time (C/D) converter*
```math
x[n] = x_{c}(n T_{s}),
```
where $T_{s}$ is the *sampling period* and has units of seconds/sample, and its inverse $f_{s} = 1 / T_s$ is the *sampling frequency* and has units of samples/second.
The sampling frequency is sometimes needed in units of radians/second, which can be obtained using the conversion
```math
\Omega_{s} = 2\pi f_{s},
```
where the dimensional analysis is `[rad/sec] = [2π rad/sample][samples/sec]`.

The DTFT is given by
```math
X_{2\pi}(\omega)
= \sum_{n=-\infty}^{\infty}x[n] e^{-\jmath\omega n},
```
where $\omega$ has units of `rad/sample`, or it can be given by
```math
X_{f_{s}}(\xi) = 
\sum_{n=-\infty}^{\infty}x[n] e^{-\jmath2\pi\xi/f_{s} n},
```
where $\xi$ has units of Hz.

The relation between the two is $X_{f_{s}}(\xi) = X_{2\pi}(2\pi \xi/f_{s})$ and the relation between the frequencies is
```math
\omega = 2\pi \xi / f_{s},
```
where the dimensional analysis is `[rad/sample] = [2π rad/cycle] [cycle / sec] / [sample/sec]`.

Note that the frequencies in the Fourier transform are different than those in the  Discrete-Time Fourier Transform, which are in [normalized units](https://en.wikipedia.org/wiki/Normalized_frequency_(signal_processing)), which have units of `radians/sample`.
"""

# ╔═╡ 02d0664d-1ff9-4938-8027-6248cec09280
md"""
# Discrete Fourier transform (DFT)
The DTFT transforms a discrete-time signal $x[n]$ for $k\in\mathbb{Z}$ into a continuous-time transform $X(\xi)$ for $\xi\in\mathbb{R}$.
On the other hand, the [discrete Fourier transform](https://en.wikipedia.org/wiki/Discrete_Fourier_transform) (DFT) transforms a discrete-time signal $x[n]$ for $n\in\mathbb{Z}$ into a discrete-time signal $X[k]$ for $k\in\mathbb{Z}$.
The DFT is the basis to the [fast Fourier transform](https://en.wikipedia.org/wiki/Fast_Fourier_transform) (FFT) algorithm and sometimes the two names are used interchangeably.

Note that similar to the DTFT, the frequenceis of the DFT (and equivalently, FFT) have units of rad/sample.
It is then possible to convert the units to (radians/sec) using the sampling frequency.

The DFT is given by
```math
X[k]
= \sum_{n=0}^{N - 1}x[n]e^{-\jmath \frac{2\pi k}{N} n},
```
where $N$ is the length of the sequence $x[n]$ and $2\pi k / N$ is the normalized frequencies (i.e., has units of rad/sample).
This frequency is an integer multiple of the *fundamental frequency* $2\pi/N$ associated with the periodic sequence $x[n]$.
Furthermore, similar to the DTFT, the DFT has a spectral (i.e., on frequency domain) periodicity of $2\pi$.
This is why the frequencies go only from $0$ to $2\pi$.
"""

# ╔═╡ 884db842-f6bb-4fc3-be7d-93a157f9ff73
md"""
## Practical considerations
Applying the DFT, or FFT algorithm, is not always straightforward and requires some "tweaking" to get the expected results.
To demonstrate these tweaks, an example is presented, which is taken from [this video](https://www.youtube.com/watch?v=mkGsMWi_j4Q&t=484s&ab_channel=SimonXu).

Let a sequence be given by
```math
x[n] = \sin(2\pi t_n) = \sin(2\pi n / f_{s}), \quad n=0,\ldots,7,
```
"""

# ╔═╡ dac0aac7-a4aa-4c7f-8ba6-6127958718da
plot(sin.(2pi * (0:7)./8))

# ╔═╡ b46c0d2f-8e17-4880-be07-264e282e7a88
LinRange(0, 1, 9)[1:end-1]

# ╔═╡ 6699be3b-284a-405f-8906-4048c2facb99
[i for i in 0:1/8:1-1/8]

# ╔═╡ 3b3f404f-6d9a-4654-9349-55a2aa4a42b4
md"""
# References
- [YouTube video explaining DFT](https://www.youtube.com/watch?v=mkGsMWi_j4Q&t=484s&ab_channel=SimonXu)
- [Answer about scaling DFT](https://dsp.stackexchange.com/questions/55047/why-do-dft-frequency-buckets-need-to-be-divided-by-sample-period)
"""

# ╔═╡ 87bf9f1b-b8d7-467c-bc97-37306cc48cf6
md"""
# Example
"""

# ╔═╡ bbb139db-5a98-468b-a1a9-e574b0eb2e9c
# Sampling frequency
fₛ = 20;

# ╔═╡ 7ddcc56f-fcf7-47c9-84fc-a71d74d559f0
# Sampling period
Tₛ = 1 / fₛ;

# ╔═╡ 0132655b-c770-4c80-9a8f-e6f09fd7bcb1
# Signal duration
# A long duration is required for the `psd` function to work
t_end = 50; # Seconds

# ╔═╡ 4f4788f8-b82b-4ad1-add5-3b0e2fd1749b
begin
	# Number of samples
  	N = ceil(t_end * fₛ);

	# Half the number of samples
	N_half = ceil(Int, N / 2);
end

# ╔═╡ 15af6853-511e-41d7-af6d-caa4e854ecc0
# Time stamps
t = LinRange(0, t_end - Tₛ, N);

# ╔═╡ 5564dbf2-9d25-47f0-97f3-20444ec718fb
md"""
Generate a sine wave function that has two underlying frequencies.
"""

# ╔═╡ 82f68cd5-0003-4fdc-a7bc-dc143136f5c9
# Generated signal
begin
	ω₁ = 1; # Hz
	x₁ = sin.(2π * ω₁ * t);
	ω₂ = 2; # Hz
	x₂ = 3 * sin.(2π * ω₂ * t);
	x = x₁ + x₂
end;

# ╔═╡ 92dd67b3-928e-4d13-8a6a-43dfae3a6db4
begin
	plot(t, x);
	xlabel!("Time (s)")
	ylabel!("Amplitude")
end

# ╔═╡ 30034f9c-35dd-4b03-b8b3-d0bfc97b135b
md"""
The dicrete Fourier transform (DFT), which is the basis for the FFT algorith, is given by
```math
X[k] = \sum_{n=1}^{N} x[n] \cdot e^{-\jmath \omega_{k} n},
```
where $\omega_{k} = 2\pi k / N \in [-\pi, \pi]$.
"""

# ╔═╡ 8d549d89-49eb-42a1-8e3f-66f1ff13efa8
X_dft(k::Int64) = sum(x .* exp.(-im * 2π * k / N * (0:N-1)));

# ╔═╡ 488ecabe-2de3-4d87-9f2e-47880b0d3c6d
X_dft_all = X_dft.(0:N-1);

# ╔═╡ 6fa1e2bd-e54d-42ba-a77e-88ba78c1e6c2
md"""
Note that the Fourier transform is symmetric across $ω = N/2$.
As such, use only the first half of the Fourier transform.
Furthermore, to preserve the signal energy, multiply the signal by 2 (to account for the removed part).
"""

# ╔═╡ 28b6d5c1-aab8-4f25-9cae-1d296b35fe72
X_one_sided = 2 * X_dft.(0:Int64(ceil(N/2)) - 1);

# ╔═╡ e2d399c0-3e17-4f24-889e-97690b953cfd
X_one_abs = abs.(X_one_sided) / N

# ╔═╡ d47117b2-bb64-479f-9b42-cdcc9461eb65
freqs = fₛ / N * (0:(N_half-1))

# ╔═╡ 3dbf3fcc-abb7-4804-b89d-ceac9f4d0e1a
md"""
## Note on frequencies
The normalized frequencies $\omega_{k} = \frac{2\pi k}{N}$ have units of [rad/sample].
To get the temporal frequencies $f_{k}$ (or $\xi_{k}$) with units of [Hz = cycles/sec], the following conversion is used

```math
\omega_{k} = 2\pi \frac{f}{f_{s}},
```
where $f_{s}$ is the sampling frequency and has units of [samples/sec].

To convert the normalized frequencies to temporal, the conversion is
```math
f = \omega_{k}\frac{1}{2\pi}f_{s} = \frac{2\pi k}{N}\frac{1}{2\pi}f_{s} = f_{s}\frac{k}{N}.
```
"""

# ╔═╡ ba43e337-223f-4a43-a1f7-29744505c1e5
length(X_one_abs)

# ╔═╡ 204c5871-f3d2-4065-9d9d-ff95a555dcdb
begin
	plot(freqs, X_one_abs)
	xlabel!("Frequency (Hz)")
	ylabel!("Amplitude")
end

# ╔═╡ 943beb7b-52ba-4ba7-a454-9674c489f0ee
angle(X_one_sided[51])

# ╔═╡ 29b0f7c0-d572-4145-ac33-e2ffdbe7ac21
md"""
## Comparison with FFT
"""

# ╔═╡ 902e9013-434a-4a28-8928-c2b9c995344b
md"""
The fast Fourier transform (FFT) *is* an efficient DFT *algorithm*.
"""

# ╔═╡ f4de1e25-7645-4d92-ae22-2e56b7c335be
X_fft = fft(x);

# ╔═╡ 7a927a47-026f-4d0a-8420-6d6e28c1ff97
X_fft

# ╔═╡ d62adbe6-1fa3-4885-9874-4517f3488b74
X_dft_all

# ╔═╡ 37628804-75e7-4904-9cf4-0899145ed824
X_fft ≈ X_dft_all

# ╔═╡ aa696b54-4ed9-4dac-9566-ab166f8f866b
plot(fₛ / N * (0:(N_half-1)), abs.(X_fft[1:(N_half)]))

# ╔═╡ e494ffa4-76cc-4a39-960f-d71d2b9672a5
md"""
## DTFT
The discrete-time Fourier transform (DTFT) takes a discrete-time signal $x[n]$ for $n\in\mathbb{Z}$ and returns a continuous-time transform $X(\omega)$, where $\omega\in[-\pi, \pi]$.
The DTFT is given by
```math
X(\omega)
=
\sum_{n=1}^{N} x[n] \cdot e^{-\jmath \omega n},
```
where $\omega \in [-\pi, \pi]$.

To convert a continuous-time frequency $\Omega_{\mathrm{ct}}$ (in Hz or cycle/sec) to discrete-time frequency (in rad/sample) $\omega$, do the following transformation
```math
\omega = 2\pi \frac{\Omega_{\mathrm{ct}}}{f_{\mathrm{s}}},
```
where $2\pi$ has units of rad/cycle and the sampling frequency $f_{\mathrm{s}}$ has units of (sample/sec).

Furthermore, note that the DFT is a specific instance of DTFT, where the discrete frequencies are given by
```math
\omega_{k}
= \frac{2\pi k}{N}.
```
"""

# ╔═╡ c542eaa9-c356-4700-82b8-c6a06d492c3b
md"""
The DTFT of the given signal $x[n]$ is
"""

# ╔═╡ ea379ef4-962b-41d3-aac6-7a668cba0a35
X_dtft(ω) = sum(x .* exp.(-im * ω * (0:N-1)));

# ╔═╡ 57b51928-7bf0-4f76-845a-11a9af8f6f31
md"""
Convert the continuous-time frequencies to discrete-time frequencies.
"""

# ╔═╡ ffa88308-1582-4586-86b5-a564b791cbab
# CT frequency to DT frequency
freq_ct_2_dt(Ω, fₛ) = 2π / fₛ * Ω;	

# ╔═╡ 12204e0b-4441-49b1-af33-bbf1862227fd
# Frequencies to generate DTFT for
begin
	freqs_ct = 0:0.01:50;
	freqs_dt = freq_ct_2_dt.(freqs_ct, fₛ);
	freqs_dt_half = freqs_dt[1:ceil(Int, length(freqs_dt)/2)]
	freqs_ct_half = freqs_ct[1:ceil(Int, length(freqs_dt)/2)]
end

# ╔═╡ 87776e85-d4c4-4a6b-a28f-a44a824b38b8
begin
	X_cts = X_dtft.(freqs_dt);
	X_cts_abs = abs.(X_cts);
	X_cts_abs_half = 2 * X_cts_abs[1:ceil(Int, length(X_cts_abs)/2)];
end;

# ╔═╡ 0e2f4d7f-409c-4cd8-a84a-ed167838d310
begin
 	plot(freqs_ct_half, X_cts_abs_half);
	xlabel!("Frequency (Hz)");
	ylabel!("Amplitude");
end

# ╔═╡ db3ca9a9-768f-48c0-86c8-200e9a2b7ea5
md"""
# Power Spectral Density (PSD)
"""

# ╔═╡ 492ff2cb-18d9-4180-85d6-6024191f30c8
md"""
### Disclaimer about the PSD above plot.
In the `psd` plot below, it seems that the x-axis represents the frequency as a fraction of the sampling frequency, which implies that the value is unitless, not in Hz as the written on the x-axis.
For example, for a sampling frequency of $f_{s}=100$ Hz, if the signal is periodic with frequency of $f=10$ Hz, then the `psd` function will show a peak at the normalized frequency $f_{\mathrm{norm}} = f / f_{s} = 10 / 100 = 0.1$, which is unitless.

What's interesting is that the `psd` function is *not* passed the sampling frequency $f_{s}$, and instead it's figuring it out on its own.
I'm not quire sure how that's done.

**Update**: It seems that the `psd` function computes frequency *per sample*, rather than in Hz.
"""

# ╔═╡ 98e779c8-3485-4bb1-a6a4-75a9a4a6e454
begin
	psd(x);
	xlabel!("Frequency (1/sample)");
end

# ╔═╡ d1ccadb1-3f77-4b36-94e8-3996d3e0a9d2
# Compute inner product of the frequencies
X_psd = [X ⋅ X for X in X_dtft.(freqs_dt)]

# ╔═╡ dcf5d6c0-f6b4-4a67-b4d4-4d262a01d1b2
begin
	plot(freqs_ct, 20log10.(abs.(X_psd) / N), ylims=(0, 100), xlims=(0, 25))
	xlabel!("Frequency (Hz)");
	ylabel!("Power");
end

# ╔═╡ 6f6c7d3d-c79a-4f83-9f6e-8e87a107e064
length(X_dtft.(freqs_dt))

# ╔═╡ f7fda800-25ec-48d2-91bf-291cd2f38cdf
plot(freqs_ct / fₛ, 20log10.(abs.(X_psd) / N), ylims=(0, 100), xlims=(0, 0.5))

# ╔═╡ 1c5b4275-503a-4c4d-9ffd-7b267e01c9a3
md"""
# White Noise
"""

# ╔═╡ e98c2abc-6eac-4950-affa-6be5bfd70558
md"""
A wide sense stationary (WSS) random process is said to be white noise if it has a constant mean
```math
\mu(t)
=
\mathbb{E}
\left[
\underline{\mathbf{x}}(t)
\right]
=
\eta
```
and its autocovariance function depends on the time difference only
```math
\operatorname{cov}
\left[
	\underline{\mathbf{x}}(t_{1}),
	\underline{\mathbf{x}}(t_{2})
\right]
=
\operatorname{cov}
\left[
\underline{\mathbf{x}}(\tau)
\right]
=
\boldsymbol{\Sigma}
\
\delta(\tau),
```
where $\tau = t_{2} - t_{1}$, and $\mathbf{\Sigma}$ is the *power spectral density (PSD)* of the signal.

The PSD matrix is the Fourier transform of the autocovariance function, so we'll try to show this here.
"""

# ╔═╡ a20af87a-2fd4-4c78-b65b-ac7ab5c56dbe
md"""
First, construct a white-noise signal using the `randn` function.
Assume a sampling rate of 1 Hz.
"""

# ╔═╡ 3f368dd9-a188-425f-964b-9a86f1a7828d
begin
	σ = 7;   # Standard deviation
	μ = 7;   # Mean    
    x_ns_offset = μ .+ σ * randn(10000);
	x_ns = x_ns_offset .- mean(x_ns_offset);
end;

# ╔═╡ 044c3a0b-1c83-4c6b-af68-3791929cbbd5
plot(x_ns)

# ╔═╡ 78c75f12-410f-4969-8a54-3cbc81621aae
md"""
Get the PSD
"""

# ╔═╡ 062523d1-b8cd-4c6f-8bf0-975965a30bb7
psd(x_ns)

# ╔═╡ 4f245438-a2c8-43c8-b60d-d29078c12437
X_fft_ns = fft(x_ns);

# ╔═╡ 1b078a10-f5a5-44ec-9d9f-13d4a9fa1ae8
X_mag_ns = [X ⋅ X for X in X_fft_ns];

# ╔═╡ 7b5da501-b5b7-49e8-a68b-d7fc238fe660
σ_est = sqrt(mean([X ⋅ X for X in fft(x_ns)]) / length(x_ns))

# ╔═╡ 748e0032-a6db-4d6f-877d-6104f7ce1973
σ - σ_est

# ╔═╡ 33b2d009-a637-4c32-b145-056e60338b80
md"""
# Pixel 3 Datasheet
Pixel 3 uses [BMI160 BOSCH](https://www.mouser.com/datasheet/2/783/BST-BMI160-DS000-1509569.pdf) accelerometer, which has an output noise of 180-300 $\mu$g/$\sqrt{\mathrm{Hz}}$.
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
SignalAnalysis = "df1fea92-c066-49dd-8b36-eace3378ea47"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[compat]
FFTW = "~1.4.6"
Plots = "~1.29.0"
PlutoUI = "~0.7.39"
SignalAnalysis = "~0.4.1"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.7.2"
manifest_format = "2.0"

[[deps.AbstractFFTs]]
deps = ["ChainRulesCore", "LinearAlgebra"]
git-tree-sha1 = "6f1d9bc1c08f9f4a8fa92e3ea3cb50153a1b40d4"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.1.0"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

[[deps.Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "af92965fb30777147966f58acb05da51c5616b5f"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.3.3"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

[[deps.Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "9950387274246d08af38f6eef8cb5480862a435f"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.14.0"

[[deps.ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "1e315e3f4b0b7ce40feded39c73049692126cf53"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.3"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "Random"]
git-tree-sha1 = "7297381ccb5df764549818d9a7d57e45f1057d30"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.18.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "a985dc37e357a3b22b260a5def99f3530fb415d3"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.2"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "SpecialFunctions", "Statistics", "TensorCore"]
git-tree-sha1 = "3f1f500312161f1ae067abe07d13b40f78f32e07"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.9.8"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[deps.Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "b153278a25dd42c65abbf4e62344f9d22e59191b"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.43.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[deps.ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f74e9d5388b8620b4cee35d4c5a618dd4dc547f4"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.3.0"

[[deps.Contour]]
deps = ["StaticArrays"]
git-tree-sha1 = "9f02045d934dc030edad45944ea80dbd1f0ebea7"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.5.7"

[[deps.DSP]]
deps = ["Compat", "FFTW", "IterTools", "LinearAlgebra", "Polynomials", "Random", "Reexport", "SpecialFunctions", "Statistics"]
git-tree-sha1 = "3e03979d16275ed5d9078d50327332c546e24e68"
uuid = "717857b8-e6f2-59f4-9121-6e50c889abd2"
version = "0.7.5"

[[deps.DataAPI]]
git-tree-sha1 = "fb5f5316dd3fd4c5e7c30a24d50643b73e37cd40"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.10.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "cc1a8e22627f33c789ab60b36a9132ac050bbf75"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.12"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[deps.DensityInterface]]
deps = ["InverseFunctions", "Test"]
git-tree-sha1 = "80c3e8639e3353e5d2912fb3a1916b8455e2494b"
uuid = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
version = "0.4.0"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Distributions]]
deps = ["ChainRulesCore", "DensityInterface", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "Test"]
git-tree-sha1 = "8a6b49396a4058771c5c072239b2e0a76e2e898c"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.58"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "b19534d1895d702889b219c382a6e18010797f0b"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.6"

[[deps.Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[deps.DualNumbers]]
deps = ["Calculus", "NaNMath", "SpecialFunctions"]
git-tree-sha1 = "5837a837389fccf076445fce071c8ddaea35a566"
uuid = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74"
version = "0.6.8"

[[deps.EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3f3a2501fa7236e9b911e0f7a588c657e822bb6d"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.3+0"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bad72f730e9e91c08d9427d5e8db95478a3c323d"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.4.8+0"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "Pkg", "Zlib_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "d8a578692e3077ac998b50c0217dfd67f21d1e5f"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.0+0"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "505876577b5481e50d089c1c68899dfb6faebc62"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.4.6"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c6033cc3892d0ef5bb9cd29b7f2f0331ea5184ea"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+0"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "9267e5f50b0e12fdfd5a2455534345c4cf2c7f7a"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.14.0"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "246621d23d1f43e3b9c368bf3b72b2331a27c286"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.13.2"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[deps.Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "87eb71354d8ec1a96d4a7636bd57a7347dde3ef9"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.4+0"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "51d2dfe8e590fbd74e7a842cf6d13d8a2f45dc01"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.6+0"

[[deps.GR]]
deps = ["Base64", "DelimitedFiles", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Printf", "Random", "RelocatableFolders", "Serialization", "Sockets", "Test", "UUIDs"]
git-tree-sha1 = "af237c08bda486b74318c8070adb96efa6952530"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.64.2"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Pkg", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "cd6efcf9dc746b06709df14e462f0a3fe0786b1e"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.64.2+0"

[[deps.GeometryBasics]]
deps = ["EarCut_jll", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "83ea630384a13fc4f002b77690bc0afeb4255ac9"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.2"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "a32d672ac2c967f3deb8a81d828afc739c838a06"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.68.3+2"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "Dates", "IniFile", "Logging", "MbedTLS", "NetworkOptions", "Sockets", "URIs"]
git-tree-sha1 = "0fa77022fe4b511826b39c894c90daf5fce3334a"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "0.9.17"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[deps.HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "SpecialFunctions", "Test"]
git-tree-sha1 = "cb7099a0109939f16a4d3b572ba8396b1f6c7c31"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.10"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "c47c5fa4c5308f27ccaac35504858d8914e102f9"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.4"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

[[deps.IniFile]]
git-tree-sha1 = "f550e6e32074c939295eb5ea6de31849ac2c9625"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.1"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d979e54b71da82f3a65b62553da4fc3d18c9004c"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2018.0.3+2"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "336cc738f03e069ef2cac55a104eb823455dca75"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.4"

[[deps.IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[deps.IterTools]]
git-tree-sha1 = "fa6287a4469f5e048d763df38279ee729fbd44e5"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.4.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "3c837543ddb02250ef42f4738347454f95079d4e"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.3"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b53380851c6e6664204efb2e62cd24fa5c47e4ba"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.2+0"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bf36f528eec6634efc60d7ec062008f171071434"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "3.0.0+1"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[deps.Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "Printf", "Requires"]
git-tree-sha1 = "46a39b9c58749eefb5f2dc1178cb8fab5332b1ab"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.15"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "7739f837d6447403596a75d19ed01fd08d6f56bf"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.3.0+3"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "42b62845d70a619f063a7da093d995ec8e15e778"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+1"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "c9551dd26e31ab17b86cbd00c2ede019c08758eb"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.3.0+1"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "09e4b894ce6a976c354a69041a04748180d43637"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.15"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "e595b205efd49508358f7dc670a940c790204629"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2022.0.0+0"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "3d3e902b31198a27340d0bf00d6ac452866021cf"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.9"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "Random", "Sockets"]
git-tree-sha1 = "1c38e51c3d08ef2278062ebceade0e46cefc96fe"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.0.3"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[deps.Measures]]
git-tree-sha1 = "e498ddeee6f9fdb4551ce855a46f54dbd900245f"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.1"

[[deps.MetaArrays]]
deps = ["Requires"]
git-tree-sha1 = "6647f7d45a9153162d6561957405c12088caf537"
uuid = "36b8f3f0-b776-11e8-061f-1f20094e1fc8"
version = "0.2.10"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[deps.MutableArithmetics]]
deps = ["LinearAlgebra", "SparseArrays", "Test"]
git-tree-sha1 = "4050cd02756970414dab13b55d55ae1826b19008"
uuid = "d8a4904e-b15c-11e9-3269-09a3773c0cb0"
version = "1.0.2"

[[deps.NaNMath]]
git-tree-sha1 = "737a5957f387b17e74d4ad2f440eb330b39a62c5"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.0"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[deps.OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "e6c5f47ba51b734a4e264d7183b6750aec459fa0"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.11.1"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ab05aa4cc89736e95915b01e7279e61b1bfe33b8"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.14+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[deps.PCRE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b2a7af664e098055a7529ad1a900ded962bca488"
uuid = "2f80f16e-611a-54ab-bc61-aa92de5b98fc"
version = "8.44.0+0"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "027185efff6be268abbaf30cfd53ca9b59e3c857"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.10"

[[deps.PaddedViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "03a7a85b76381a3d04c7a1656039197e70eda03d"
uuid = "5432bcbf-9aad-5242-b902-cca2824c8663"
version = "0.5.11"

[[deps.Parsers]]
deps = ["Dates"]
git-tree-sha1 = "1285416549ccfcdf0c50d4997a94331e88d68413"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.3.1"

[[deps.Pixman_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b4f5d02549a10e20780a24fce72bea96b6329e29"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.40.1+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "8162b2f8547bc23876edd0c5181b27702ae58dce"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.0.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "bb16469fd5224100e422f0b027d26c5a25de1200"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.2.0"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "GeometryBasics", "JSON", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun", "Unzip"]
git-tree-sha1 = "d457f881ea56bbfa18222642de51e0abf67b9027"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.29.0"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "UUIDs"]
git-tree-sha1 = "8d1f54886b9037091edf146b517989fc4a09efec"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.39"

[[deps.Polynomials]]
deps = ["LinearAlgebra", "MutableArithmetics", "RecipesBase"]
git-tree-sha1 = "ee0cfbea3d8a44f677d59f5df4677889c4d71846"
uuid = "f27b6e38-b328-58d1-80ce-0feddd5e7a45"
version = "3.0.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "47e5f437cc0e7ef2ce8406ce1e7e24d44915f88d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.3.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "c6c0f690d0cc7caddb74cef7aa847b824a16b256"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+1"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "78aadffb3efd2155af139781b8a8df1ef279ea39"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.4.2"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.RecipesBase]]
git-tree-sha1 = "6bf3f380ff52ce0832ddd3a2a7b9538ed1bcca7d"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.2.1"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "RecipesBase"]
git-tree-sha1 = "dc1e451e15d90347a7decc4221842a022b011714"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.5.2"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "cdbd3b1338c72ce29d9584fdbe9e9b70eeb5adca"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "0.1.3"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "bf3188feca147ce108c76ad82c2792c57abe7b1f"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.0"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "68db32dff12bb6127bac73c209881191bf0efbb7"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.3.0+0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "0b4b7f1393cff97c33891da2a0bf69c6ed241fda"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.1.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SignalAnalysis]]
deps = ["DSP", "Distributions", "DocStringExtensions", "FFTW", "LinearAlgebra", "MetaArrays", "PaddedViews", "Random", "Requires", "SignalBase", "Statistics", "WAV"]
git-tree-sha1 = "da5e232a8fe1f08d047f3d1614ad9a25e439a390"
uuid = "df1fea92-c066-49dd-8b36-eace3378ea47"
version = "0.4.1"

[[deps.SignalBase]]
deps = ["Unitful"]
git-tree-sha1 = "14cb05cba5cc89d15e6098e7bb41dcef2606a10a"
uuid = "00c44e92-20f5-44bc-8f45-a1dcef76ba38"
version = "0.1.2"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "bc40f042cfcc56230f781d92db71f0e21496dffd"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.1.5"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "cd56bf18ed715e8b09f06ef8c6b781e6cdc49911"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.4.4"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "c82aaa13b44ea00134f8c9c89819477bd3986ecd"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.3.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "8977b17906b0a1cc74ab2e3a05faa16cf08a8291"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.16"

[[deps.StatsFuns]]
deps = ["ChainRulesCore", "HypergeometricFunctions", "InverseFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "5783b877201a82fc0014cbf381e7e6eb130473a4"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.0.1"

[[deps.StructArrays]]
deps = ["Adapt", "DataAPI", "StaticArrays", "Tables"]
git-tree-sha1 = "e75d82493681dfd884a357952bbd7ab0608e1dc3"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.7"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits", "Test"]
git-tree-sha1 = "5ce79ce186cc678bbb5c5681ca3379d1ddae11a1"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.7.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.Tricks]]
git-tree-sha1 = "6bac775f2d42a611cdfcd1fb217ee719630c4175"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.6"

[[deps.URIs]]
git-tree-sha1 = "97bbe755a53fe859669cd907f2d96aee8d2c1355"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.3.0"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unitful]]
deps = ["ConstructionBase", "Dates", "LinearAlgebra", "Random"]
git-tree-sha1 = "b649200e887a487468b71821e2644382699f1b0f"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.11.0"

[[deps.Unzip]]
git-tree-sha1 = "34db80951901073501137bdbc3d5a8e7bbd06670"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.1.2"

[[deps.WAV]]
deps = ["Base64", "FileIO", "Libdl", "Logging"]
git-tree-sha1 = "7e7e1b4686995aaf4ecaaf52f6cd824fa6bd6aa5"
uuid = "8149f6b0-98f6-5db9-b78f-408fbbb8ef88"
version = "1.2.0"

[[deps.Wayland_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "3e61f0b86f90dacb0bc0e73a0c5a83f6a8636e23"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.19.0+0"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4528479aa01ee1b3b4cd0e6faef0e04cf16466da"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.25.0+0"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "1acf5bdf07aa0907e0a37d3718bb88d4b687b74a"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.9.12+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "5be649d550f3f4b95308bf0183b82e2582876527"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.6.9+4"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4e490d5c960c314f33885790ed410ff3a94ce67e"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.9+4"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fe47bd2247248125c428978740e18a681372dd4"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.3+4"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6783737e45d3c59a4a4c4091f5f88cdcf0908cbb"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.0+3"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "daf17f441228e7a3833846cd048892861cff16d6"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.13.0+3"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "926af861744212db0eb001d9e40b5d16292080b2"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.0+4"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "4bcbf660f6c2e714f87e960a171b119d06ee163b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.2+4"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "5c8424f8a67c3f2209646d4425f3d415fee5931d"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.27.0+4"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "79c31e7844f6ecf779705fbc12146eb190b7d845"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.4.0+3"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e45044cd873ded54b6a5bac0eb5c971392cf1927"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.2+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "ece2350174195bb31de1a63bea3a41ae1aa593b6"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "0.9.1+5"
"""

# ╔═╡ Cell order:
# ╟─d9160695-c11a-4e69-94fc-26e4302e08e4
# ╟─831f8744-d659-11ec-3d14-090c4a016610
# ╟─5dec98a0-95b1-4cf5-a71f-27f3d4ac8139
# ╟─f4a89cf7-6c0e-43b3-b09b-81669d9c280d
# ╟─24a64d23-2a86-41a9-9350-760838bcb186
# ╟─02d0664d-1ff9-4938-8027-6248cec09280
# ╟─884db842-f6bb-4fc3-be7d-93a157f9ff73
# ╠═dac0aac7-a4aa-4c7f-8ba6-6127958718da
# ╠═b46c0d2f-8e17-4880-be07-264e282e7a88
# ╠═6699be3b-284a-405f-8906-4048c2facb99
# ╟─3b3f404f-6d9a-4654-9349-55a2aa4a42b4
# ╟─87bf9f1b-b8d7-467c-bc97-37306cc48cf6
# ╠═f31d07b1-d65d-47f6-b23c-fd7c0b58a506
# ╠═bbb139db-5a98-468b-a1a9-e574b0eb2e9c
# ╠═7ddcc56f-fcf7-47c9-84fc-a71d74d559f0
# ╠═0132655b-c770-4c80-9a8f-e6f09fd7bcb1
# ╠═4f4788f8-b82b-4ad1-add5-3b0e2fd1749b
# ╠═15af6853-511e-41d7-af6d-caa4e854ecc0
# ╟─5564dbf2-9d25-47f0-97f3-20444ec718fb
# ╠═82f68cd5-0003-4fdc-a7bc-dc143136f5c9
# ╠═92dd67b3-928e-4d13-8a6a-43dfae3a6db4
# ╟─30034f9c-35dd-4b03-b8b3-d0bfc97b135b
# ╠═8d549d89-49eb-42a1-8e3f-66f1ff13efa8
# ╠═488ecabe-2de3-4d87-9f2e-47880b0d3c6d
# ╟─6fa1e2bd-e54d-42ba-a77e-88ba78c1e6c2
# ╠═28b6d5c1-aab8-4f25-9cae-1d296b35fe72
# ╠═e2d399c0-3e17-4f24-889e-97690b953cfd
# ╠═d47117b2-bb64-479f-9b42-cdcc9461eb65
# ╟─3dbf3fcc-abb7-4804-b89d-ceac9f4d0e1a
# ╠═ba43e337-223f-4a43-a1f7-29744505c1e5
# ╠═204c5871-f3d2-4065-9d9d-ff95a555dcdb
# ╠═943beb7b-52ba-4ba7-a454-9674c489f0ee
# ╟─29b0f7c0-d572-4145-ac33-e2ffdbe7ac21
# ╟─902e9013-434a-4a28-8928-c2b9c995344b
# ╠═f4de1e25-7645-4d92-ae22-2e56b7c335be
# ╠═7a927a47-026f-4d0a-8420-6d6e28c1ff97
# ╠═d62adbe6-1fa3-4885-9874-4517f3488b74
# ╠═37628804-75e7-4904-9cf4-0899145ed824
# ╠═aa696b54-4ed9-4dac-9566-ab166f8f866b
# ╟─e494ffa4-76cc-4a39-960f-d71d2b9672a5
# ╟─c542eaa9-c356-4700-82b8-c6a06d492c3b
# ╠═ea379ef4-962b-41d3-aac6-7a668cba0a35
# ╟─57b51928-7bf0-4f76-845a-11a9af8f6f31
# ╠═ffa88308-1582-4586-86b5-a564b791cbab
# ╠═12204e0b-4441-49b1-af33-bbf1862227fd
# ╠═87776e85-d4c4-4a6b-a28f-a44a824b38b8
# ╠═0e2f4d7f-409c-4cd8-a84a-ed167838d310
# ╟─db3ca9a9-768f-48c0-86c8-200e9a2b7ea5
# ╠═492ff2cb-18d9-4180-85d6-6024191f30c8
# ╠═98e779c8-3485-4bb1-a6a4-75a9a4a6e454
# ╠═d1ccadb1-3f77-4b36-94e8-3996d3e0a9d2
# ╠═dcf5d6c0-f6b4-4a67-b4d4-4d262a01d1b2
# ╠═6f6c7d3d-c79a-4f83-9f6e-8e87a107e064
# ╠═f7fda800-25ec-48d2-91bf-291cd2f38cdf
# ╟─1c5b4275-503a-4c4d-9ffd-7b267e01c9a3
# ╟─e98c2abc-6eac-4950-affa-6be5bfd70558
# ╟─a20af87a-2fd4-4c78-b65b-ac7ab5c56dbe
# ╠═3f368dd9-a188-425f-964b-9a86f1a7828d
# ╠═044c3a0b-1c83-4c6b-af68-3791929cbbd5
# ╟─78c75f12-410f-4969-8a54-3cbc81621aae
# ╠═062523d1-b8cd-4c6f-8bf0-975965a30bb7
# ╠═4f245438-a2c8-43c8-b60d-d29078c12437
# ╠═1b078a10-f5a5-44ec-9d9f-13d4a9fa1ae8
# ╠═7b5da501-b5b7-49e8-a68b-d7fc238fe660
# ╠═748e0032-a6db-4d6f-877d-6104f7ce1973
# ╟─33b2d009-a637-4c32-b145-056e60338b80
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
