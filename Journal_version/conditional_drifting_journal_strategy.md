# Strategy File: Strengthening the Conditional Drifting Channel Simulation Paper

## Working title

**Condition-Wise Sinkhorn Drifting for One-Shot Learned Channel Simulation**

Alternative titles:

- **Fiber-Preserving Sinkhorn Drifting for Low-Latency Learned Channel Simulation**
- **One-Shot Conditional Transport Surrogates for Learned Channel Simulation**
- **Condition-Wise Wasserstein Drifting for Learned Channel Simulation**

Recommended title: **Condition-Wise Sinkhorn Drifting for One-Shot Learned Channel Simulation**

Reason: it foregrounds the actual contribution: not merely using drifting, but making drifting valid for conditional channel laws through condition-wise/fiber-wise Sinkhorn transport.

---

## 1. Core positioning

The paper should not be positioned as:

> We compare drifting, diffusion, DDIM, WGAN, residual drifting, direct drifting, joint kernels, and Sinkhorn variants.

That reads like a broad benchmark with many moving parts.

The stronger positioning is:

> Diffusion-based channel simulators are accurate but iterative. Drifting offers one-shot sampling, but existing drifting/W-Flow methods are unconditional and therefore not directly valid for channel simulation. Channel simulation requires preserving the transmitted input \(x\) and transporting only the conditional output law \(p(y\mid x)\). We formulate condition-wise Sinkhorn drifting, a fiber-preserving one-shot channel surrogate, and show that it gives a useful latency–accuracy tradeoff for learned communication-system training.

The central claim should be:

> **One-shot transport surrogates for communication channels must operate on conditional fibers \(p(y\mid x)\), not on the global output cloud.**

Everything else should support this.

---

## 2. Main novelty statement

Use this novelty statement in the introduction, abstract, and response to reviewers:

> Existing diffusion-based learned channel simulators model \(p(y\mid x)\) through iterative reverse denoising. They provide strong sample quality but inherit multi-step inference. Existing drifting and W-Flow methods offer one-shot generation, but they are formulated for unconditional target distributions and do not address the fixed-input structure of channel laws. In communication-channel simulation, the transmitted symbol \(x\) is side information and must not be transported. The proposed condition-wise Sinkhorn drifting instead transports only the output law over each fixed condition, yielding a fiber-preserving one-shot surrogate for \(p(y\mid x)\).

Keep the claim conservative:

- Do **not** claim that drifting dominates diffusion.
- Do claim that condition-wise drifting gives a practically useful one-shot alternative.
- Do claim that naive/global transport is the wrong geometry for conditional channel simulation.
- Do claim that global output-space SWD is insufficient for selecting a channel surrogate.

---

## 3. Target venue strategy

### Primary target: IEEE Transactions on Communications (TCOM)

Best fit if the paper remains method-driven and broadly about learned channel simulation.

Why TCOM fits:

- Broad communications scope.
- Channel simulation and coding/autoencoder checks are relevant.
- The contribution is not tied to a specific wireless architecture.

Estimated acceptance probability after strengthening:

\[
P(\text{eventual TCOM acceptance}) \approx 0.35\text{--}0.50.
\]

### Secondary target: IEEE Transactions on Machine Learning in Communications and Networking (TMLCN)

Possibly the strongest topical fit if the paper is framed as ML-for-communications.

Estimated acceptance probability:

\[
P(\text{eventual TMLCN acceptance}) \approx 0.40\text{--}0.55.
\]

### TWC only after wireless strengthening

IEEE Transactions on Wireless Communications is possible only if the wireless part is upgraded substantially.

Needed additions for TWC:

- realistic 3GPP TDL/CDL/MIMO-OFDM setting,
- mobility or frequency selectivity,
- measured/ray-tracing channel data,
- link-level BLER/EVM/rate results,
- explicit wireless system endpoint.

Without this, the paper may be judged as a generative-modeling/channel-simulation paper rather than a wireless-communications paper.

---

## 4. Most important revision package

If only a limited revision is possible, do these four things.

### 4.1 Add a conditional-fiber diagnostic figure

Purpose: show why global SWD is insufficient and why condition-wise transport matters.

Recommended figure:

For several fixed anchors \(x_i\), plot the conditional output clouds:

\[
\{y_{i,j}\}_{j=1}^{K} \sim p(y\mid x_i),
\qquad
\{\hat y_{i,j}\}_{j=1}^{K} \sim q_\theta(y\mid x_i).
\]

Compare:

1. analytic channel,
2. naive/global drifting or joint Sinkhorn,
3. condition-wise Sinkhorn.

Possible diagnostics:

\[
x \mapsto \mathbb{E}[\hat y\mid x],
\]

\[
x \mapsto \operatorname{Cov}(\hat y\mid x),
\]

or anchor-wise conditional SWD/GW2.

Main message:

> A model can achieve acceptable global output-space SWD while matching the wrong conditional fibers. Downstream SER follows the conditional-fiber quality rather than the global output-cloud metric.

This is probably the highest-value addition.

---

### 4.2 Add equal wall-clock autoencoder-training comparison

Purpose: turn the latency advantage into a communication-system argument.

Experiment:

Train the same symbolic autoencoder or TurboAE through different learned channel implants under an equal wall-clock budget.

Compare:

- analytic channel,
- DDPM or DDIM implant,
- condition-wise Sinkhorn implant,
- WGAN if available.

Report:

- wall-clock time,
- number of channel calls,
- epochs/updates completed,
- final SER/BER/BLER under analytic-channel evaluation.

Possible table:

| Implant | Sampling steps | Channel calls/sec | Training time | AE updates | Final SER/BER |
|---|---:|---:|---:|---:|---:|
| Analytic | 1 | high | fixed | fixed | floor |
| DDPM | 100 | low | fixed | fewer | strong but expensive |
| DDIM-20 | 20 | medium | fixed | fewer | quality-speed tradeoff |
| Condition-wise Sinkhorn | 1 | high | fixed | full | close to analytic/diffusion |
| WGAN | 1 | high | fixed | full | unstable/variable |

Main message:

> For inner-loop learned communication-system training, one-shot surrogates can be preferable even when diffusion has better standalone sample quality.

---

### 4.3 Add local conditional Sinkhorn proof-of-concept

Purpose: address the main limitation: repeated samples at exactly the same \(x\).

Current exact condition-wise Sinkhorn assumes simulator access:

\[
y_{i,j} \sim p(\cdot\mid x_i), \qquad j=1,\ldots,K.
\]

For measured channels, one may only have one observation per condition:

\[
(x_i,y_i)_{i=1}^N.
\]

Add a local conditional approximation:

\[
\hat p_h(y\mid x_0)
=
\sum_{i=1}^N
\alpha_i(x_0)\delta_{y_i},
\]

with

\[
\alpha_i(x_0)
=
\frac{K_h(x_i,x_0)}
{\sum_{\ell=1}^N K_h(x_\ell,x_0)}.
\]

Then solve a weighted local Sinkhorn problem around \(x_0\).

Recommended experiment:

Simulate a dataset where each \(x_i\) has only one \(y_i\). Compare:

1. exact repeated-condition Sinkhorn,
2. local conditional Sinkhorn,
3. global/joint Sinkhorn,
4. naive drifting.

Main message:

> The exact condition-wise estimator is natural for simulator channels. For measured datasets, local conditional Sinkhorn provides a practical approximation and remains closer to the correct conditional law than global transport.

This closes a likely reviewer objection.

---

### 4.4 Rewrite the introduction around one central claim

New introduction structure:

1. Learned communication systems need differentiable stochastic channel surrogates.
2. Diffusion models are accurate but iterative.
3. One-shot generators are attractive for inner-loop training.
4. Drifting/W-Flow is one-shot but currently unconditional.
5. Channel simulation is conditional: \(x\) is fixed side information.
6. Therefore transport must be fiber-wise over \(p(y\mid x)\).
7. Proposed method: condition-wise Sinkhorn drifting.
8. Contributions and evidence.

Avoid a laundry list of model variants in the introduction.

---

## 5. Theory strengthening

The theory should be precise but modest.

### Population objective

Define the conditional Sinkhorn functional:

\[
S_\varepsilon^{\mathrm{cond}}(q_\theta,p)
=
\int S_\varepsilon(q_{\theta,x},p_x)\,\mu(dx),
\]

where

\[
p(dx,dy)=\mu(dx)p_x(dy),
\qquad
q_\theta(dx,dy)=\mu(dx)q_{\theta,x}(dy).
\]

### Population flow statement

Under regularity assumptions for the Sinkhorn potentials and finite second moments, the exact conditional Wasserstein flow satisfies

\[
\frac{d}{dt}
S_\varepsilon^{\mathrm{cond}}(q_t,p)
=
-
\int
\|v_t(x,\cdot)\|^2_{L^2(q_{t,x})}
\,\mu(dx)
\le 0.
\]

If the condition-wise Sinkhorn velocity has equality only when

\[
q_{t,x}=p_x
\quad
\text{for } \mu\text{-a.e. }x,
\]

then stationary points of the population conditional flow correspond to equality of conditional channel laws.

### Important caveat

Explicitly separate three levels:

1. **Population conditional OT flow**
   - Mathematical object.
   - Has the fiber-wise equilibrium interpretation.

2. **Finite-sample Sinkhorn particle estimator**
   - Approximates the population velocity with minibatch samples.

3. **Projected neural generator training**
   - Uses detached-target regression.
   - Does not equal exact gradient descent on \(S_\varepsilon^{\mathrm{cond}}\).
   - Converges only to the best representable surrogate under the projected training dynamics.

This distinction makes the theory harder to attack.

Suggested phrasing:

> We do not claim convergence of the finite-width neural generator to the exact conditional Sinkhorn flow. The population result only identifies the correct geometry and equilibrium condition for channel simulation. The implemented algorithm is a particle approximation followed by projection into the generator class through detached-target regression.

---

## 6. Experiments to add or improve

### 6.1 Conditional mean/covariance diagnostics

For each anchor \(x_i\), estimate:

\[
m_p(x_i)=\mathbb{E}[y\mid x_i],
\qquad
m_q(x_i)=\mathbb{E}[\hat y\mid x_i],
\]

and

\[
C_p(x_i)=\operatorname{Cov}(y\mid x_i),
\qquad
C_q(x_i)=\operatorname{Cov}(\hat y\mid x_i).
\]

Metrics:

\[
\frac{1}{N_a}
\sum_{i=1}^{N_a}
\|m_p(x_i)-m_q(x_i)\|_2^2,
\]

\[
\frac{1}{N_a}
\sum_{i=1}^{N_a}
\|C_p(x_i)-C_q(x_i)\|_F^2.
\]

Add to anchor diagnostics table or appendix.

### 6.2 SER/BER versus global SWD scatter plot

Plot each surrogate as a point:

\[
\text{x-axis: global direct-output SWD},
\qquad
\text{y-axis: downstream SER}.
\]

Expected message:

> Low global SWD does not guarantee low downstream SER.

This directly supports the paper's metric critique.

### 6.3 Equal-inference-budget diffusion comparison

Compare:

- DDIM-10,
- DDIM-20,
- DDIM-50,
- condition-wise Sinkhorn,
- WGAN.

Instead of only reporting sample time and SWD separately, show an accuracy-latency Pareto curve:

\[
(\text{inference time per sample}, \text{SER/BER or SWD}).
\]

Main message:

> Diffusion gives the best high-quality endpoint, but condition-wise Sinkhorn occupies a useful low-latency region of the Pareto frontier.

### 6.4 More realistic TDL/CDL experiment if targeting TWC

If time allows, add one stronger wireless experiment:

- longer complex block,
- 3GPP TDL/CDL with OFDM,
- multiple taps without aggressive rounding,
- possibly MIMO,
- link-level BLER.

For TCOM this is optional. For TWC it is close to mandatory.

---

## 7. Variants to keep in main paper versus appendix

The current draft has many variants. This can dilute the story.

### Main paper should keep

- DDPM/DDIM reference,
- WGAN reference,
- naive direct drifting,
- joint/global Sinkhorn,
- condition-wise Sinkhorn.

### Move to appendix or compress

- residual drifting,
- target-kernel drifting,
- joint-kernel drifting,
- extra SWD-only tables,
- extensive hyperparameter details.

Main paper story:

1. diffusion is accurate but iterative,
2. drifting is one-shot but unconditional,
3. naive/global transport fails conditional geometry,
4. condition-wise Sinkhorn fixes the geometry,
5. one-shot surrogate is useful for learned communication-system training.

---

## 8. Abstract rewrite target

The abstract should contain these exact ideas:

1. channel simulators are called repeatedly inside learned communication-system training;
2. diffusion models are accurate but require iterative reverse denoising;
3. drifting gives one-shot generation but existing drifting/W-Flow is unconditional;
4. communication channels require preserving \(x\) and matching \(p(y\mid x)\);
5. propose condition-wise Sinkhorn drifting;
6. show strong latency advantage and competitive downstream performance;
7. acknowledge diffusion remains strongest on hardest SER curves.

Possible abstract skeleton:

> Accurate stochastic channel surrogates are useful in learned communication systems, where the channel model may be evaluated millions of times inside differentiable training loops. Conditional diffusion models provide strong sample quality, but their inference cost scales with the number of reverse denoising steps. This paper studies drifting models as one-shot learned channel simulators. We show that a direct application of unconditional drifting or W-Flow is not geometrically aligned with channel simulation: the transmitted input \(x\) is fixed side information, and only the conditional output law \(p(y\mid x)\) should be transported. We therefore formulate condition-wise Sinkhorn drifting, which computes Sinkhorn/Wasserstein drift fields separately on the output fiber over each transmitted symbol. Experiments on AWGN, Rayleigh, nonlinear amplifier, and compact TDL channels compare the proposed method with DDPM, DDIM, WGAN, and drifting ablations. The results show that condition-wise Sinkhorn is the strongest one-shot drifting-family surrogate under conditional diagnostics and downstream coding checks, while diffusion remains the strongest learned reference on the hardest SER curves. The method provides a low-latency, differentiable, condition-preserving alternative for inner-loop channel simulation.

---

## 9. Introduction paragraph to add

Suggested paragraph:

> The distinction between unconditional generation and channel simulation is central. In ordinary generative modeling, samples are transported in the data space until the model distribution matches the target distribution. In channel simulation, however, the transmitted symbol is not part of the stochastic object to be transported. It is side information. The relevant target is the family of conditional laws \(\{p(\cdot\mid x)\}_{x\sim\mu}\). A generator that matches the global output cloud can still mix conditional fibers and produce an incorrect channel surrogate. This motivates a condition-wise transport formulation in which couplings preserve \(x\) and move only the output component \(y\). The proposed condition-wise Sinkhorn drifting implements this idea as a one-shot neural channel generator.

---

## 10. Discussion paragraph to add

Suggested discussion paragraph:

> The proposed method should be interpreted as a low-latency conditional surrogate rather than a universal replacement for diffusion. The experiments show that diffusion remains the strongest learned reference on several hard downstream SER curves. However, diffusion pays for this accuracy through iterative sampling. Condition-wise Sinkhorn drifting targets a different operating point: one generator evaluation, differentiability, and condition-preserving stochasticity. This is especially relevant when the learned channel is embedded in an inner training loop, where millions of surrogate calls can dominate the computational budget. The main methodological lesson is that for channel surrogates, the geometry of the conditional law matters more than global sample-cloud agreement.

---

## 11. Reviewer-risk checklist

### Risk: “This is just applying drifting to channels.”

Response:

> Existing drifting/W-Flow is unconditional. A direct application does not respect the fixed-input structure of channel laws. The paper contributes the condition-wise/fiber-preserving formulation and demonstrates empirically that naive/global variants can fail conditional diagnostics and downstream SER.

### Risk: “Diffusion performs better.”

Response:

> The paper does not claim that condition-wise drifting dominates diffusion. Diffusion remains the strongest high-quality learned reference on harder SER curves. The contribution is a one-shot, low-latency, condition-preserving surrogate that occupies a different point on the accuracy-latency frontier.

### Risk: “The comparisons are not equal-capacity.”

Response:

> The diffusion/WGAN rows are reference baselines. The controlled comparison is the matched-architecture drift-field ablation. Claims about superiority are restricted to the drifting-family variants. Cross-family comparisons are interpreted as practical accuracy-latency references, not strict capacity-controlled rankings.

### Risk: “Repeated samples at the same \(x\) are unrealistic.”

Response:

> This is exact for simulator-style channels and differentiable synthetic channels. For measured data, the paper proposes or evaluates local conditional Sinkhorn, replacing exact repeated samples with kernel-weighted neighborhoods in condition space.

### Risk: “Global SWD already looks good.”

Response:

> Global SWD does not guarantee correct conditional laws. The paper reports anchor-conditioned diagnostics and downstream SER/BER to show that conditional-fiber quality is the relevant criterion.

---

## 12. Recommended final paper structure

1. **Introduction**
   - motivation,
   - diffusion latency,
   - one-shot drifting opportunity,
   - conditional-fiber issue,
   - contributions.

2. **Background**
   - learned channel simulation,
   - diffusion/DDIM,
   - drifting/W-Flow,
   - why unconditional transport is insufficient.

3. **Condition-Wise Sinkhorn Drifting**
   - channel law factorization,
   - conditional Sinkhorn objective,
   - population flow,
   - finite-sample estimator,
   - neural detached-target training.

4. **Experimental Setup**
   - channels,
   - models,
   - metrics,
   - timing protocol.

5. **Results**
   - global SWD benchmark,
   - conditional diagnostics,
   - downstream SER/BER,
   - accuracy-latency Pareto,
   - equal wall-clock training-loop experiment.

6. **Measured-Data Extension**
   - local conditional Sinkhorn proof-of-concept,
   - limitations.

7. **Discussion**
   - diffusion vs one-shot surrogate tradeoff,
   - metric selection,
   - limitations.

8. **Conclusion**

Appendix:

- residual drifting,
- extra kernel variants,
- hyperparameter details,
- additional seed tables,
- implementation details.

---

## 13. Minimal action plan before submission

### Must do

- Rewrite intro around conditional fibers.
- Add figure showing conditional-fiber mismatch.
- Add or strengthen accuracy-latency Pareto.
- Move weaker variants to appendix or compress them.
- Make all cross-family claims conservative.

### Strongly recommended

- Equal wall-clock autoencoder-training experiment.
- Local conditional Sinkhorn approximation for one-sample-per-\(x\) data.
- Scatter plot of global SWD versus downstream SER.

### Optional

- More realistic TDL/CDL/OFDM/MIMO experiment.
- Measured or ray-tracing channel data.
- Additional theory proposition with formal assumptions.

---

## 14. One-sentence paper identity

Use this sentence to keep the revision focused:

> **This paper makes one-shot drifting applicable to learned channel simulation by replacing unconditional/global transport with condition-wise Sinkhorn transport over the fibers \(p(y\mid x)\), yielding a low-latency differentiable surrogate whose usefulness is best judged by conditional diagnostics and downstream communication metrics rather than global SWD alone.**
