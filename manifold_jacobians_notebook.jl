### A Pluto.jl notebook ###
# v0.19.5

using Markdown
using InteractiveUtils

# ╔═╡ bff69758-b72f-11ec-302f-d381893b0229
begin	
	using LinearAlgebra
	using ForwardDiff	
	using ShortCodes
	using SparseArrays
	using PlutoUI
end

# ╔═╡ c6010a83-3fd2-4f24-88de-6ebd685e78ca
PlutoUI.TableOfContents(title="📚 Table of Contents", indent=true, depth=4, aside=true)

# ╔═╡ 24395b9c-ab7e-40fb-8438-032d34e03ca0
md"""
In this notebook aims at understanding left and right Jacobians of a (Lie group) manifold.
The notebook uses theory from the references listed at the end of the document.
"""

# ╔═╡ 8332f1c2-7b68-4fb4-acab-cf9838259796
md"""
#  Parametrizing matrix Lie group elements
"""

# ╔═╡ a16ccdde-0b0c-4647-a1a6-20835be22f5c
md"""
Given a manifold $G$ with $n$ degrees of freedom, then an element $\mathbf{X}\in G \subset \mathbb{R}^{m\times m}$ can be parameterized by a set of parameters $\boldsymbol{\xi}=\begin{bmatrix}\xi_{1} & \xi_{2} & \ldots & \xi_{n}\end{bmatrix}^{\mathsf{T}}\in\mathbb{R}^{n}$, where $n$ is known as the *degrees of freedom* of the group and, *in general*, $n\neq m$.

There are different ways of parametrization.
For example, consider the planar *special euclidean* matrix Lie group $SE(2)$, where the elements $\mathbf{T}\in SE(2)$, known as *transformation matrices*, are given by
```math
\begin{align}
\mathbf{T}
&=
\begin{bmatrix}
	\mathbf{C} & \mathbf{r} \\
 	\mathbf{0} & 1
\end{bmatrix}
=
\begin{bmatrix}
	\cos(\theta) & -\sin(\theta) & r_{1} \\
 	\sin(\theta) & \cos(\theta) & r_{2} \\
 		0 		 & 		0 		& 1
\end{bmatrix}
\in SE(2) \subset \mathbb{R}^{3\times 3},
\end{align}
```
where $\mathbf{r}\in\mathbb{R}^{2}$ are the *translation* components, $\mathbf{C}\in SO(2)$ is a *rotation matrix*, also referred to as a *cosine direction matrix* (DCM) hence the letter $\mathbf{C}$, and $SO(2)$ is the planar *special orthogonal* matrix Lie group.

The only manifolds discussed in this document are matrix Lie groups.
Therefore, the terms *matrix Lie group*, *Lie group*, *group*, *manifold* will be used interchangeably.
"""

# ╔═╡ 343cba3c-99f6-4254-8774-6e3c4643bf96
md"""
## The translation-rotation parametrization
"""

# ╔═╡ 4b74769d-b2dc-4712-82b1-624df4effca2
md"""
The transformation matrix $\mathbf{T}$ can be parameterized in multiple ways.
For example, one parametrization, known as translation-rotation (T-R) parametrization [3], is given by
```math
\mathbf{T}
=
\exp(r_{1}\mathbf{E}_{1} + r_{2}\mathbf{E}_{2})\exp(\theta\mathbf{E}_{3}),
```
where
$\mathbf{E}_{i}$ are the *generators* (or the *basis*) of the Lie algebra for the given parameterization, and for $\mathfrak{se}(2)$ are given by
```math
\begin{align}
\mathbf{E}_{1}
=
\begin{bmatrix}
0 & 0 & 1 \\
0 & 0 & 0 \\
0 & 0 & 0 
\end{bmatrix},
\;
\mathbf{E}_{2}
=
\begin{bmatrix}
0 & 0 & 0 \\
0 & 0 & 1 \\
0 & 0 & 0 
\end{bmatrix},
\;
\mathbf{E}_{3}
=
\begin{bmatrix}
0 & -1 & 0 \\
1 & 0 & 0 \\
0 & 0 & 0 
\end{bmatrix}.
\end{align}
```
The T-R parameters can be concatendated into a single bold-faced parameter
```math
\boldsymbol{\xi}^{\mathsf{TR}}
=
\begin{bmatrix}
r_{1} & r_{2} & \theta
\end{bmatrix}^{\mathsf{T}}
\in \mathbb{R}^{3}
```
"""

# ╔═╡ e10d15e8-97c5-4313-80f6-691f141d4d2a
md"Let's code up the generator matrices"

# ╔═╡ 97a7c195-a3dd-4b0a-a040-2891a6b3a50c
# Generators
begin	
	E = [];
	push!(E, collect(sparse([1], [3], 1, 3, 3)));
	push!(E, collect(sparse([2], [3], 1, 3, 3)));
	push!(E, collect(sparse([1, 2], [2, 1], [-1, 1], 3, 3)));
end

# ╔═╡ 84d96324-1c4e-4176-95c8-6435dcff5980
md"""
Note that the `collect` function is used to convert the sparse matrices into dense matrices.
This is because some of the functions defined below do not operate with sparse matrices.
Or at least, I couldn't figure out how to do so.
"""

# ╔═╡ 942d716a-0725-4a7c-993b-d77f519135ab
md"""
# The exponential parametrization
"""

# ╔═╡ bd588eec-eb97-4504-8295-356068caa843
md"""
Another popular parameterization is the *exponential coordinates* parameterization, where the generators $\mathbf{E}_{i}$ are the *exactly as above*.
The tranformation matrix is parametrized as
```math
\mathbf{T}
=
\exp\left(\sum_{i=1}^{3}\xi_{i}\mathbf{E}_{i}\right)
=
\exp\left(\boldsymbol{\xi}^{\wedge}\right),
```
where the $(\cdot)^{\wedge}$ is the *wedge* operator described below and the parameters $\boldsymbol{\xi}$, sometimes referred to as the *exponential parameters*, are
```math
\boldsymbol{\xi}
=
\begin{bmatrix}
\rho_{1} & \rho_{2} & \theta
\end{bmatrix}^{\mathsf{T}}
\in \mathbb{R}^{3}.
```
At a first glance, the two parameters $\boldsymbol{\xi}^{\mathsf{TR}}$ and $\boldsymbol{\xi}$ look identical, but they are *not*!
Actually, only $\theta$ is the same in both, but $r_{i}\neq\rho_{i}$, in general.
Well, how are the two parametrizations related? That's *one* of many applications of the manifold Jacobians that will be discussed in this notebook.

In this notebook, and in many applications, the exponential coordinates are used.
"""

# ╔═╡ db21cac0-2e0b-4c58-bc0c-b9412ce7eb0d
md"""
To simplify notation, the *wedge* $(\cdot)^{\wedge}:\mathbb{R}^{n}\to\mathfrak{g}\subset\mathbb{R}^{m\times m}$ and *vee* $(\cdot)^{\vee}:\mathfrak{g}\subset\mathbb{R}^{m\times m}\to\mathbb{R}^{n}$ operators can be used.
Specifically, the operators are given by
```math
\boldsymbol{\Xi}
:=
\boldsymbol{\xi}^{\wedge}
=
\sum_{i=1}^{n}\xi_{i}\mathbf{E}_{i}
=
\begin{bmatrix}
0 & -\theta & \rho_{1} \\
\theta & 0 & \rho_{2} \\ 
0 & 0 & 1
\end{bmatrix}
\in\mathfrak{se}(2)\subset\mathbb{R}^{3\times 3},
```
and
```math
\boldsymbol{\xi}
:=
\boldsymbol{\Xi}^{\vee}
=
\begin{bmatrix}
\rho_{1} & \rho_{2} & \theta
\end{bmatrix}^{\mathsf{T}}
\in\mathbb{R}^{n}
```
"""

# ╔═╡ cee87f2e-0a07-4dd4-baf6-88cb8d0f7892
md"""
Let's code up the wedge and vee operators.
"""

# ╔═╡ 7433e66f-af05-4c9e-8c32-f42bdc6fbb4b
wedge(ξ) = sum(E .* ξ);

# ╔═╡ 2462929a-3665-458b-beff-8327ecb294e9
vee(Ξ) = [Ξ[1, 3]; Ξ[2, 3]; Ξ[2, 1]];

# ╔═╡ f7ef8e24-e951-448a-8e09-3f49d7122fa6
md"""
The vee function assumes the "proper" matrices with the $\mathfrak{se}(2)$ structure are passed to the function. Thus, the function does not check the structure of the matrices.
In practice, a type for the $\mathfrak{se}(2)$ should be defined.
"""

# ╔═╡ ad218788-1359-4f88-8c83-92de42839a4a
md"""
It is rare to use the Lie algebra elements when working with Lie groups since they are matrices, and they can be mapped from the Euclidean space using the wedge and vee operators.
Instead, the Lie group elements $\mathbf{X}(\boldsymbol{\xi})\in G$ and their parameters $\boldsymbol{\xi}\in\mathbb{R}^{n}$ are used.

The reason for using the Lie group elements is to *preserve the structure* of the manifold (e.g., preserving rotations), whereas the parameters are used because they live in a *Euclidean linear* space (i.e., they are "vectors" or *column matrices*), which is *much* easier to work with than matrix spaces.

As such, the mappings $\operatorname{Exp}:\mathbb{R}^{n}\to G\subset\mathbb{R}^{m\times m}$ and its inverse $\operatorname{Log}:G\subset\mathbb{R}^{m\times m}\to\mathbb{R}^{n}$ are defined to simplify notation, and are given by
```math
\operatorname{Exp}(\boldsymbol{\xi})
\overset{\mathsf{def}}{=}
\exp\left(\boldsymbol{\xi}^{\wedge}\right),
\quad
\operatorname{Log}(\mathbf{X})
\overset{\mathsf{def}}{=}
\log\left(\mathbf{X}\right)^{\vee}.
```
"""

# ╔═╡ ae55cefb-f237-46dd-ac0c-5278a5dbd202
md"""
Let's code up the $\operatorname{Exp}$ and $\operatorname{Log}$ functions
"""

# ╔═╡ 16e4a8f7-80a4-4c23-8e48-5c9c69cb9306
Exp(ξ) = exp(wedge(ξ));

# ╔═╡ ca76714c-c656-4786-b583-50514399bcdc
Log(X) = vee(log(X));

# ╔═╡ 729f87d9-acfa-45b7-ac87-d508f9b30b42
md"""
# The matrix Lie group constraints
"""

# ╔═╡ 4997739d-744a-4a2a-85c2-913daaa52c08
md"""
The special thing about matrix Lie groups is that it's a set of matrices closed under multiplication.
Furthermore, a *group* (any group, not only matrix Lie group) must have an identity element and an inverse.
This implied that elements of matrix Lie groups must satisfy
```math
\mathbf{X}^{-1}\mathbf{X}
=
\mathbf{X}\mathbf{X}^{-1}
=
\mathbf{1},
```
where $\mathbf{1}\in G$ is the *identity* element of the group.
"""

# ╔═╡ 9daccd2a-722c-4074-ba38-401510316e51
md"""
## The tangent space and the Lie algebra
"""

# ╔═╡ 45238b62-2bb6-4d08-b8a5-6e07654207b7
md"""
The *Lie algebra* $\mathfrak{g}$ of a group $G$ is the *tangent space at the identity element of the group*.
Therefore, to understand the Lie algebra we need to understand the tangent space.

Consdier a trajectory on the manifold given by $\mathbf{X}(t)\in G$.
Then the time derivative of this trajectory is in the *tangent space* of $G$ *at $\mathbf{X}$*, which is denoted by
```math
\dot{\mathbf{X}}(t) \in T_{\mathbf{X}(t)}G.
```
The tangent space is then defined as the set of *all possible time derivative of all trajectories on $G$ passing through $\mathbf{X}$*.
"""

# ╔═╡ 91e97df5-571f-4234-9abc-c757cc84d1d3
md"""
Matrix Lie groups are smooth manifolds, and smooth manifolds have the same *structure* of the tangent space *everywhere*.
Therefore, vectors in tangent space at $\mathbf{X}\in G$, denoted by $\dot{\mathbf{X}}\in T_{\mathbf{X}}G$ can be mapped to the tangent space at $\mathbf{YX}\in G$ by premultiplying the tangent vector with $\mathbf{Y}\in G$
```math
\mathbf{Y}\dot{\mathbf{X}} \in T_{\mathbf{Y}\mathbf{X}}G.
```
Similarly, to map the tangent vector at $\mathbf{X}\in G$ to the tangent space at $\mathbf{X}\mathbf{Y}$, postmultiply the tangent vector with $\mathbf{Y}\in G$
```math
\dot{\mathbf{X}}\mathbf{Y}\in T_{\mathbf{X}\mathbf{Y}}G.
```
"""
# Need to work on this more and cite references

# ╔═╡ 7e5d503e-adc7-464a-bde7-229c2004848b
md"""
In the special case that $\mathbf{Y}:=\mathbf{X}^{-1}$, then the tangent vectors are mapped to the tangent space at the *identity*, which is the *Lie algebra*!
That is,
```math
\mathbf{X}^{-1}\dot{\mathbf{X}} \in T_{\mathbf{X}^{-1}\mathbf{X}}G = T_{\mathbf{1}}G = \mathfrak{g}.
```
Similarly,
```math
\dot{\mathbf{X}}\mathbf{X}^{-1} \in T_{\mathbf{X}\mathbf{X}^{-1}}G = T_{\mathbf{1}}G = \mathfrak{g}.
```
"""

# ╔═╡ 0af5ec2f-0529-4f6f-9f42-64dff3cdc1d0
md"""
The Lie algebra elements can be written as
```math
\mathbf{v}_{r}^{\wedge}
:=
\mathbf{X}^{-1}\dot{\mathbf{X}},
\quad
\mathbf{v}_{\ell}^{\wedge}
:=
\dot{\mathbf{X}}\mathbf{X}^{-1}.
```

But... what do the superscripts $(\cdot)_{r}$ and $(\cdot)_{\ell}$ mean?

To answer the question, set $\dot{\mathbf{X}}$ as the argument in the above equations.
This results in
```math
\dot{\mathbf{X}} = \mathbf{X} \mathbf{v}_{r}^{\wedge},
\quad
\dot{\mathbf{X}} = \mathbf{v}_{\ell}^{\wedge}\mathbf{X},
```
where it is clear that $\mathbf{v}_{r}^{\wedge}$ is applied on the *right*, whereas $\mathbf{v}_{\ell}^{\wedge}$ is applied on the *left*.
Sola [1] has a good overview on the interpretation of these vectors.

The equation above also shows how the two vectors are related.
Specifically,
```math
\mathbf{v}_{\ell}^{\wedge} = \mathbf{X}\mathbf{v}_{r}^{\wedge}\mathbf{X}^{-1},
```
which can also be written as
```math
\mathbf{v}_{\ell}
= \left(\mathbf{X}\mathbf{v}_{r}^{\wedge}\mathbf{X}^{-1}\right)^{-1}
= \operatorname{Adj}_{\mathbf{X}}\mathbf{v}_{r},
```
where $\operatorname{Adj}$ is the *Adjoint* operator [1-3].
"""

# ╔═╡ 0338c314-ee8f-4051-8734-7613e0bdb0a3
md"""
# Left and right manifold Jacobians
"""

# ╔═╡ 42b121ec-4619-475a-9798-81f03035d993
md"""
In the last section, the tangent space and the Lie algebra were introduced as spaces *tangent to the manifold* at some point.
Therefore, tangent vectors, including Lie algebra vectors, can be thought of as "velocity" vectors of the manifold.

Additionally, recall that the matrix Lie groups are parametrized by $\boldsymbol{\xi}$.
Therefore, a natural question pops: *what is the relation between the time-rate of change of the parameters, $\dot{\boldsymbol{\xi}}$ and the tangent vectors $\mathbf{v}_{r}$ and $\mathbf{v}_{\ell}$?*

This is described by the Jacobians.
Furthermore, since there are two types of tangent vectors, $\mathbf{v}_{r}$ and $\mathbf{v}_{\ell}$, then there will be two types of Jacobians: a *right Jacobian* $\mathbf{J}_{r}$ that associates $\dot{\boldsymbol{\xi}}$ with $\mathbf{v}_{r}$, and a *left Jacobian* $\mathbf{J}_{\ell}$ that associates $\dot{\boldsymbol{\xi}}$ with $\mathbf{v}_{l}$.

Specifically, the Jacobians are given by [3]
```math
\mathbf{v}_{r}
=\mathbf{J}_{r}(\boldsymbol{\xi})\dot{\boldsymbol{\xi}},
\quad
\mathbf{v}_{\ell}
=\mathbf{J}_{\ell}(\boldsymbol{\xi})\dot{\boldsymbol{\xi}},
```
"""

# ╔═╡ 6bc30514-cb45-4986-95a9-3d000710ea32
md"""
## Deriving the Jacobians
"""

# ╔═╡ f24b35ee-db24-46a2-9bc0-aff9d3ded6dd
md"""
**Note**: The left and right Jacobians are derived in the same manner.
Therefore, only the right Jacobian is derived in this section but the same theory is applied to derive the left Jacobian.
"""

# ╔═╡ c25659a3-bd65-43bc-8ef2-a0cd03bb635f
md"""

As discussed the previous sections, Lie group elements $\mathbf{X}$ can be described using different parametrizations $\boldsymbol{\xi}$.
Therefor, the time-rate of change of the parameter $\boldsymbol{\xi}$ can be derived from differentiating $\mathbf{X}$.

Specifically,
```math
\dot{\mathbf{X}}
=
\frac{\partial{\left(\mathbf{X}(\boldsymbol{\xi})\right)}}{\partial t}
=
\sum_{i=1}^{n}
\frac{\partial \mathbf{X}(\boldsymbol{\xi})}{\partial \xi_{i}}
\dot{\xi}_{i}
```
"""

# ╔═╡ 54370e2d-1b3e-407c-8ded-864abb7d9d4f
md"""
Inserting the above equation into the (right) Lie algebra/group constraint from the *Tangent space and Lie algebra* section results in
```math
\mathbf{v}_{r}
=
\left(
\mathbf{X}^{-1}\dot{\mathbf{X}}
\right)^{\vee}
=
\left(
\mathbf{X}(\boldsymbol{\xi})^{-1}
\left(\sum_{i=1}^{n}
\frac{\partial \mathbf{X}(\boldsymbol{\xi})}{\partial \xi_{i}}
\dot{\xi}_{i}\right)
\right)^{\vee}
```
"""

# ╔═╡ a5f128ab-7788-451a-b904-1cd8e85ff254
md"""
It is not immediately trivial how to extract parameters $\dot{\xi}_{i}$ from the inner brackets.
This is done using inner products defined on the Lie algebra [3].
Luckily, there's another (maybe easier) way: take the Jacobian with respect to the parameters $\dot{\xi}_{i}$, which is possible due to the linearity of the vee operator $(\cdot)^{\vee}$.

Specifcially,
```math
\mathbf{J}_{\mathsf{r}}
=
\frac{\mathrm{d}\mathbf{v}}{\mathrm{d}\dot{\boldsymbol{\xi}}}
\in\mathbb{R}^{n\times n},
```
where each column of the Jacobian is given by
```math
\left[
\mathbf{J}_{\mathsf{r}}
\right]_{i}
=
\left(
\mathbf{X}(\boldsymbol{\xi})^{-1}
\left(
\frac{\partial \mathbf{X}(\boldsymbol{\xi})}{\partial \xi_{i}}
\right)
\right)^{\vee}
\in\mathbb{R}^{n}
```
"""

# ╔═╡ 05c60f0e-410b-4ffb-81b9-fcec33df328d
md"""
One important note about the Jacobians is that they are **dependent on the parameterization** of the group elements, which is denoted by $\mathbf{J}_{r}(\boldsymbol{\xi})$.
That is, different parametrizations get different Jacobians, so it's important to keep that in mind.
"""

# ╔═╡ 55b6af0b-c923-4ded-a08f-da48f98f2a0d
md"""
## Computing the right Jacobians numerically
"""

# ╔═╡ d25ff5e6-814b-4b47-ad1e-21593c8eae68
md"""
Let's try to compute the right Jacobian of the SE(2) group numerically and compare with the analytical solution.

Note that there are many derivatives to compute.
Thus, the derivatives will be approximated using finite-difference method from the [`FiniteDiff.jl` package](https://github.com/JuliaDiff/FiniteDiff.jl).
"""

# ╔═╡ 1845c777-efea-4886-b536-930c28974a7c
md"""
The `FiniteDiff.jl` package couldn't handle differentiating a matrix with respect to parameters, or simply I couldn't figure it out.
So I wrote a simple finite-difference operator that takes the partial derivative of a function with respect to *one* of its parameters.
"""

# ╔═╡ 41e9f170-823b-483e-a0f3-3e5476d5abe7
md"""
### Finite-difference operator
"""

# ╔═╡ fb49ca00-af11-435f-bcdd-0374a9829f90
finite_diff_i(i, f, x; ϵ=1e-6) = (f(x + collect(sparse([i], [1], ϵ, length(x), 1))) .- f(x)) / ϵ;

# ╔═╡ e221041b-2b25-466e-a48f-be750bcb2507
md"""
### Manifold derivative
"""

# ╔═╡ 20833d04-b886-4803-befd-ab13f2453051
md"""
Let's write a function that computes
```math
\dot{\mathbf{X}}(\boldsymbol{\xi}, \dot{\boldsymbol{\xi}})
=
\sum_{i=1}^{3}
\frac{\partial \mathbf{X}(\boldsymbol{\xi})}{\partial \xi_{i}}\dot{\xi}_{i}
```
"""

# ╔═╡ 6caa2d7f-a3c6-4fcf-9e46-d41bc3a1fecf
Ẋ(ξ, ξ̇) = sum([finite_diff_i(i, Exp, ξ) for i ∈ 1:length(ξ)] .* ξ̇);

# ╔═╡ 0cd16849-dbc9-42d8-9c9a-9af28b9d4397
md"""
### Right Lie algebra vector
"""

# ╔═╡ bea4ee58-1561-45e3-b585-a9277471de18
md"""
Set the code for computing $\mathbf{v}_{r}$
```math
\mathbf{v}_{r}(\boldsymbol{\xi}, \dot{\boldsymbol{\xi}})
=
\left(
\mathbf{X}(\boldsymbol{\xi})^{-1}\dot{\mathbf{X}}(\boldsymbol{\xi}, \dot{\boldsymbol{\xi}})
\right)^{\vee}
=
\left(
\mathbf{X}(\boldsymbol{\xi})^{-1}
\left(\sum_{i=1}^{n}
\frac{\partial \mathbf{X}(\boldsymbol{\xi})}{\partial \xi_{i}}
\dot{\xi}_{i}\right)
\right)^{\vee}
```
"""

# ╔═╡ b3da0d33-88ef-496a-b437-9802b924fbb4
vᵣ(ξ, ξ̇) = vee(Exp(ξ) \ Ẋ(ξ, ξ̇));

# ╔═╡ cdc12286-d7ba-4af8-86c8-6dd387ca728e
md"""
### The right Jacobian
"""

# ╔═╡ 63493ee3-dfeb-49d9-b8a0-1eb9bb0080d1
md"""
Finally, construct the right Jacobian using
```math
\left[
\mathbf{J}_{\mathsf{r}}(\boldsymbol{\xi})
\right]_{i}
=
\frac{\partial \mathbf{v}_{r}(\boldsymbol{\xi}, \dot{\boldsymbol{\xi}})}
{\partial \dot{\xi}_{i}}
=
\left(
\mathbf{X}(\boldsymbol{\xi})^{-1}
\left(
\frac{\partial \mathbf{X}(\boldsymbol{\xi})}{\partial \xi_{i}}
\right)
\right)^{\vee}
\in\mathbb{R}^{n}
```
"""

# ╔═╡ a7e18440-b154-4aeb-9c4a-d5cb0f36e581
Jᵣ(ξ) = ForwardDiff.jacobian(ξ̇ -> vᵣ(ξ, ξ̇), ξ);

# ╔═╡ e92d6825-80c7-49cc-92f8-d0e38cbdc86b
md"""
### Comparing with analytic solution
"""

# ╔═╡ b139b79e-0d16-4f3a-88ad-cf591d47a42f
md"""
The analytical solution to the right Jacobian is given by [1, 3]
```math
\mathbf{J}_{r}(\boldsymbol{\xi})
=
\begin{bmatrix}
\sin \theta / \theta & (1-\cos \theta) / \theta & \left(\theta \rho_{1}-\rho_{2}+\rho_{2} \cos \theta-\rho_{1} \sin \theta\right) / \theta^{2} \\
(\cos \theta-1) / \theta & \sin \theta / \theta & \left(\rho_{1}+\theta \rho_{2}-\rho_{1} \cos \theta-\rho_{2} \sin \theta\right) / \theta^{2} \\
0 & 0 & 1
\end{bmatrix}.
```
"""

# ╔═╡ fd2c06f8-a66b-4fd4-a573-e94ca8184be1
Jᵣ_true(ξ) =
	[sin(ξ[3])/ξ[3] (1 - cos(ξ[3]))/ξ[3] (ξ[3]ξ[1] - ξ[2] + ξ[2]cos(ξ[3]) - 	ξ[1]sin(ξ[3]))/ξ[3]^2;
	(cos(ξ[3])-1)/ξ[3] sin(ξ[3])/ξ[3] (ξ[1] + ξ[3]ξ[2] - ξ[1]cos(ξ[3]) - ξ[2]sin(ξ[3]))/ξ[3]^2;
	0 0 1];

# ╔═╡ 119d96ed-5632-4d4e-9da8-101e89f89ef9
md"""
Generate a random parameter $\boldsymbol{\xi}$ and compare with the numerically computed and the true Jacobians.
"""

# ╔═╡ 0bcfe95d-0c5d-40c9-aa97-4394df58b872
begin
	ξ = 10 * rand(3);
	isapprox(Jᵣ(ξ), Jᵣ_true(ξ); atol=1e-5);
end

# ╔═╡ 137fc32d-e308-42a3-8540-948279911dcf
md"""
# References
"""

# ╔═╡ dd91b631-e927-4638-aadc-4aca13acdc5e
[DOI("10.48550/ARXIV.1812.01537"), DOI("10.1017/9781316671528"), DOI("10.1007/978-0-8176-4944-9_10")]

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
ShortCodes = "f62ebe17-55c5-4640-972f-b59c0dd11ccf"
SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[compat]
ForwardDiff = "~0.10.25"
PlutoUI = "~0.7.38"
ShortCodes = "~0.3.3"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.7.2"
manifest_format = "2.0"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "9950387274246d08af38f6eef8cb5480862a435f"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.14.0"

[[deps.ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "bf98fa45a0a4cee295de98d4c1462be26345b9a1"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.2"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "ded953804d019afa9a3f98981d99b33e3db7b6da"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "024fe24d83e4a5bf5fc80501a314ce0d1aa35597"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.0"

[[deps.CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[deps.Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "96b0bc6c52df76506efc8a441c6cf1adcb1babc4"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.42.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[deps.DiffResults]]
deps = ["StaticArrays"]
git-tree-sha1 = "c18e98cba888c6c25d1c3b048e4b3380ca956805"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.0.3"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "dd933c4ef7b4c270aacd4eb88fa64c147492acf0"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.10.0"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "b19534d1895d702889b219c382a6e18010797f0b"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.6"

[[deps.Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "1bd6fc0c344fc0cbee1f42f8d2e7ec8253dda2d2"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.25"

[[deps.HTTP]]
deps = ["Base64", "Dates", "IniFile", "Logging", "MbedTLS", "NetworkOptions", "Sockets", "URIs"]
git-tree-sha1 = "0fa77022fe4b511826b39c894c90daf5fce3334a"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "0.9.17"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
git-tree-sha1 = "2b078b5a615c6c0396c77810d92ee8c6f470d238"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.3"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

[[deps.IniFile]]
git-tree-sha1 = "f550e6e32074c939295eb5ea6de31849ac2c9625"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.1"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "91b5dcf362c5add98049e6c29ee756910b03051d"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.3"

[[deps.IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

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

[[deps.JSON3]]
deps = ["Dates", "Mmap", "Parsers", "StructTypes", "UUIDs"]
git-tree-sha1 = "8c1f668b24d999fb47baf80436194fdccec65ad2"
uuid = "0f8b85d8-7281-11e9-16c2-39a750bddbf1"
version = "1.9.4"

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

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "58f25e56b706f95125dcb796f39e1fb01d913a71"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.10"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

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

[[deps.Memoize]]
deps = ["MacroTools"]
git-tree-sha1 = "2b1dfcba103de714d31c033b5dacc2e4a12c7caa"
uuid = "c03570c3-d221-55d1-a50c-7939bbd78826"
version = "0.4.4"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[deps.NaNMath]]
git-tree-sha1 = "b086b7ea07f8e38cf122f5016af580881ac914fe"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "0.3.7"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Parsers]]
deps = ["Dates"]
git-tree-sha1 = "621f4f3b4977325b9128d5fae7a8b4829a0c2222"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.2.4"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "UUIDs"]
git-tree-sha1 = "670e559e5c8e191ded66fa9ea89c97f10376bb4c"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.38"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "d3538e7f8a790dc8903519090857ef8e1283eecd"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.2.5"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.ShortCodes]]
deps = ["Base64", "CodecZlib", "HTTP", "JSON3", "Memoize", "UUIDs"]
git-tree-sha1 = "0fcc38215160e0a964e9b0f0c25dcca3b2112ad1"
uuid = "f62ebe17-55c5-4640-972f-b59c0dd11ccf"
version = "0.3.3"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "5ba658aeecaaf96923dce0da9e703bd1fe7666f9"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.1.4"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "4f6ec5d99a28e1a749559ef7dd518663c5eca3d5"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.4.3"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StructTypes]]
deps = ["Dates", "UUIDs"]
git-tree-sha1 = "d24a825a95a6d98c385001212dc9020d609f2d4f"
uuid = "856f2bd8-1eba-4b0a-8007-ebc267875bd4"
version = "1.8.1"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "216b95ea110b5972db65aa90f88d8d89dcb8851c"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.6"

[[deps.URIs]]
git-tree-sha1 = "97bbe755a53fe859669cd907f2d96aee8d2c1355"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.3.0"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
"""

# ╔═╡ Cell order:
# ╠═bff69758-b72f-11ec-302f-d381893b0229
# ╠═c6010a83-3fd2-4f24-88de-6ebd685e78ca
# ╟─24395b9c-ab7e-40fb-8438-032d34e03ca0
# ╟─8332f1c2-7b68-4fb4-acab-cf9838259796
# ╟─a16ccdde-0b0c-4647-a1a6-20835be22f5c
# ╟─343cba3c-99f6-4254-8774-6e3c4643bf96
# ╟─4b74769d-b2dc-4712-82b1-624df4effca2
# ╟─e10d15e8-97c5-4313-80f6-691f141d4d2a
# ╠═97a7c195-a3dd-4b0a-a040-2891a6b3a50c
# ╟─84d96324-1c4e-4176-95c8-6435dcff5980
# ╟─942d716a-0725-4a7c-993b-d77f519135ab
# ╟─bd588eec-eb97-4504-8295-356068caa843
# ╟─db21cac0-2e0b-4c58-bc0c-b9412ce7eb0d
# ╟─cee87f2e-0a07-4dd4-baf6-88cb8d0f7892
# ╠═7433e66f-af05-4c9e-8c32-f42bdc6fbb4b
# ╠═2462929a-3665-458b-beff-8327ecb294e9
# ╟─f7ef8e24-e951-448a-8e09-3f49d7122fa6
# ╟─ad218788-1359-4f88-8c83-92de42839a4a
# ╟─ae55cefb-f237-46dd-ac0c-5278a5dbd202
# ╠═16e4a8f7-80a4-4c23-8e48-5c9c69cb9306
# ╠═ca76714c-c656-4786-b583-50514399bcdc
# ╟─729f87d9-acfa-45b7-ac87-d508f9b30b42
# ╟─4997739d-744a-4a2a-85c2-913daaa52c08
# ╟─9daccd2a-722c-4074-ba38-401510316e51
# ╟─45238b62-2bb6-4d08-b8a5-6e07654207b7
# ╟─91e97df5-571f-4234-9abc-c757cc84d1d3
# ╟─7e5d503e-adc7-464a-bde7-229c2004848b
# ╟─0af5ec2f-0529-4f6f-9f42-64dff3cdc1d0
# ╟─0338c314-ee8f-4051-8734-7613e0bdb0a3
# ╟─42b121ec-4619-475a-9798-81f03035d993
# ╟─6bc30514-cb45-4986-95a9-3d000710ea32
# ╟─f24b35ee-db24-46a2-9bc0-aff9d3ded6dd
# ╟─c25659a3-bd65-43bc-8ef2-a0cd03bb635f
# ╟─54370e2d-1b3e-407c-8ded-864abb7d9d4f
# ╟─a5f128ab-7788-451a-b904-1cd8e85ff254
# ╟─05c60f0e-410b-4ffb-81b9-fcec33df328d
# ╟─55b6af0b-c923-4ded-a08f-da48f98f2a0d
# ╟─d25ff5e6-814b-4b47-ad1e-21593c8eae68
# ╟─1845c777-efea-4886-b536-930c28974a7c
# ╟─41e9f170-823b-483e-a0f3-3e5476d5abe7
# ╠═fb49ca00-af11-435f-bcdd-0374a9829f90
# ╟─e221041b-2b25-466e-a48f-be750bcb2507
# ╟─20833d04-b886-4803-befd-ab13f2453051
# ╠═6caa2d7f-a3c6-4fcf-9e46-d41bc3a1fecf
# ╟─0cd16849-dbc9-42d8-9c9a-9af28b9d4397
# ╟─bea4ee58-1561-45e3-b585-a9277471de18
# ╠═b3da0d33-88ef-496a-b437-9802b924fbb4
# ╟─cdc12286-d7ba-4af8-86c8-6dd387ca728e
# ╟─63493ee3-dfeb-49d9-b8a0-1eb9bb0080d1
# ╠═a7e18440-b154-4aeb-9c4a-d5cb0f36e581
# ╟─e92d6825-80c7-49cc-92f8-d0e38cbdc86b
# ╟─b139b79e-0d16-4f3a-88ad-cf591d47a42f
# ╠═fd2c06f8-a66b-4fd4-a573-e94ca8184be1
# ╟─119d96ed-5632-4d4e-9da8-101e89f89ef9
# ╠═0bcfe95d-0c5d-40c9-aa97-4394df58b872
# ╟─137fc32d-e308-42a3-8540-948279911dcf
# ╟─dd91b631-e927-4638-aadc-4aca13acdc5e
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
