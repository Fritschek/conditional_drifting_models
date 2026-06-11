# TCOM Review Comments  
**Paper:** *Condition-Wise Sinkhorn Drifting for One-Shot Learned Channel Simulation*  
**Recommendation:** Major Revision  
**Estimated probability of eventual TCOM acceptance:** 45–60%  
**Estimated probability of acceptance in current form:** 20–30%

---

## Summary

This manuscript studies learned stochastic channel simulation for differentiable communication-system training. The authors propose **condition-wise Sinkhorn drifting**, a one-shot conditional generator trained by per-condition optimal-transport-inspired particle updates.

The central idea is to avoid global transport over the joint output cloud and instead transport samples inside fixed-input conditional fibers \(p(y|x)\). The proposed method is evaluated on AWGN, Rayleigh fading, SSPA nonlinearity, and a compact TDL channel, with comparisons against conditional DDPM, DDIM, WGAN, direct drifting, kernel drifting, joint Sinkhorn drifting, and condition-wise Sinkhorn drifting.

The main empirical conclusion is that condition-wise Sinkhorn is the strongest one-shot drifting-family surrogate under conditional diagnostics and symbolic coding checks, while diffusion remains stronger on the hardest downstream SER curves.

---

## Recommendation

**Major Revision.**

The paper addresses an important and timely problem: low-latency learned channel simulation for communication-system training. The condition-wise transport viewpoint is natural and technically meaningful. The manuscript is also unusually transparent about limitations, metric dependence, and the fact that diffusion remains stronger in several downstream settings.

However, in its current form, the paper does not yet provide sufficient theoretical clarity or experimental control for acceptance in TCOM. The contribution is promising, but the paper needs a sharper formal statement of what is proved, a cleaner separation between population optimal-transport theory and the implemented detached regression algorithm, and a more convincing experimental protocol.

---

## Strengths

1. **Relevant problem formulation.**  
   The paper correctly identifies that learned channel simulators must approximate conditional laws \(p(y|x)\), not merely global output distributions. This distinction is important for communication applications and is not always handled carefully in generative-channel papers.

2. **Conceptually sound condition-wise transport idea.**  
   The proposed formulation of a conditional Sinkhorn objective over fixed-input fibers is well motivated. Restricting transport to preserve \(x\) is the right geometry for channel simulation.

3. **Good latency motivation.**  
   The paper makes a clear case for one-shot generators when the channel model is called many times inside autoencoder training, decoder adaptation, or Monte Carlo loops.

4. **Broad empirical evaluation.**  
   The manuscript includes AWGN, Rayleigh, SSPA, and TDL channels, and compares against DDPM, DDIM, WGAN, and several drifting/Sinkhorn ablations.

5. **Useful diagnostic message.**  
   The paper convincingly argues that global SWD alone is insufficient for channel-surrogate selection and that anchor-conditioned metrics and downstream SER/BER checks are more relevant.

6. **Honest positioning.**  
   The manuscript does not overclaim that the proposed method dominates diffusion. It explicitly states that diffusion remains strongest on difficult downstream SER curves.

---

## Major Comments

### 1. The theoretical contribution needs sharper boundaries

The population condition-wise Sinkhorn functional is well motivated, but the manuscript sometimes moves too quickly from the population Wasserstein-flow interpretation to the implemented neural training rule. The implemented method uses finite minibatch Sinkhorn couplings, barycentric projections, detached targets, and neural regression. This is not exact gradient descent on the stated conditional Sinkhorn functional with respect to \(\theta\).

The paper does acknowledge this, but the distinction should be made more central. In particular, the authors should state explicitly:

- what object is optimized at the population level;
- what object is approximated by the minibatch estimator;
- whether the detached-target regression is a surrogate update rather than an unbiased gradient step;
- under what conditions the stationary-point argument applies only to the population flow, not necessarily to the neural algorithm.

At present, the theory is plausible but not rigorous enough for the strength of the surrounding claims.

---

### 2. The novelty relative to W-Flow and drifting needs to be stated more precisely

The paper builds on drifting models and W-Flow/Sinkhorn-gradient-flow ideas. The main novelty appears to be the condition-wise restriction of the transport problem for channel simulation. This is a meaningful adaptation, but the manuscript should sharpen the novelty claim.

The authors should clearly separate:

- borrowed components: drifting, detached target regression, Sinkhorn/W-Flow velocity, barycentric projection;
- new components: condition-wise coupling preserving \(x\), per-anchor conditional Sinkhorn estimator, application to learned channel simulation;
- empirical contribution: showing that condition-wise transport is superior to joint/global transport for conditional channel surrogates.

This would make the contribution easier to evaluate and reduce the risk that the paper appears to repackage existing W-Flow machinery.

---

### 3. Experimental fairness is not fully convincing

The manuscript is transparent that the diffusion, WGAN, and drifting models are not strictly matched in parameter count, training budget, or tuning effort. However, this substantially limits the strength of the comparative conclusions. For TCOM, the paper should either provide a better controlled comparison or narrow the claims.

In particular:

- DDPM/DDIM, WGAN, and drifting use different architectures and training schedules.
- The number of seeds differs across experiments.
- The SSPA condition-wise Sinkhorn result uses a compact update-budget screen, while other rows use different budgets.
- Some TDL baseline rows are from checkpoint-only evaluation.
- The paper states that diffusion remains stronger on downstream curves, but the tables and figures should make the comparison more systematic.

The authors should add at least one matched-capacity or matched-wall-clock comparison for the key channels. Alternatively, they should explicitly frame diffusion/WGAN rows as contextual references and restrict the main claim to the matched drifting-family ablation.

---

### 4. The SSPA update-budget issue is a concern

The SSPA results are important because the channel is nonlinear and communication-relevant. However, the manuscript reports that the corrected same-condition Sinkhorn field saturates after fewer optimizer updates and that continuing to the full preset over-optimizes the field. This raises questions about robustness and hyperparameter sensitivity.

The authors should provide:

- training curves for SSPA condition-wise Sinkhorn;
- sensitivity to update budget, Sinkhorn \(\epsilon\), number of particles per condition, and drift scale;
- a clear model-selection rule not based on downstream test performance;
- an explanation of whether over-optimization is a general limitation of sharp condition-wise transport fields.

Without this, the SSPA result may appear hand-tuned.

---

### 5. Conditional sampling assumptions should be emphasized earlier

The method assumes simulator-style access to repeated outputs at the same transmitted symbol \(x\). This is natural for analytic/synthetic channels but much less natural for measured datasets. The manuscript discusses this limitation, but it should be elevated in the introduction or problem formulation.

The paper should clearly state that the current method is best suited for analytic, simulator, or controlled measurement settings where repeated conditional samples are available. The proposed local conditional Sinkhorn extension for measured channels is interesting but remains future work.

---

### 6. Downstream evaluation should be expanded or clarified

The symbolic coding experiments are valuable, but the downstream setup needs more explanation. In particular:

- How are the autoencoders trained?
- Are encoder and decoder architectures fixed across surrogate channels?
- Are hyperparameters selected independently of the analytic-channel test curve?
- Are the SER/BER confidence intervals computed across seeds of the surrogate, the autoencoder, or both?
- Is the analytic-channel reference a lower bound, or merely a separate training condition?

The long-block TurboAE check is useful, but it shows a nontrivial gap between analytic-channel training and condition-wise Sinkhorn training. This should be discussed more directly.

---

### 7. Some claims should be toned down

The manuscript should avoid phrasing that suggests the practical algorithm inherits the convergence properties of the population conditional Sinkhorn flow. For example, statements about equilibrium and stationarity should be carefully qualified. The correct claim is that the population conditional flow has the desired fixed-point geometry; the implemented algorithm is an approximate projected training scheme motivated by that flow.

Similarly, the phrase “best-performing one-shot surrogate” should be restricted to the evaluated drifting-family variants and the reported metrics. Since diffusion and sometimes WGAN remain competitive or superior in downstream performance, the claim should not be phrased globally.

---

## Minor Comments

1. The notation around direct-output and residual modes should be tightened. The paper sometimes moves between \(y\), \(e=y-x\), \(u\), \(g_i\), and \(\hat y\) quickly.

2. The SSPA AM/AM formula should be checked for consistency. The text describes a Rapp-style law, but the notation around \(a(r)\) and \(a(r)r\) could be confusing.

3. The term “one-shot” should be defined precisely as one neural forward pass at inference, excluding latent sampling and conditioning overhead.

4. The timing table is useful, but the comparison between training time and inference time should be interpreted carefully. For many communication tasks, the amortization point depends on the number of surrogate calls.

5. The paper should include pseudocode with all practical hyperparameters: number of anchors, generated samples per anchor, positive samples per anchor, reference samples, Sinkhorn iterations, \(\epsilon\), drift clipping, and optimizer settings.

6. The figures are useful, especially the fixed-condition SSPA visualization, but the captions should state sample sizes and whether the displayed samples are from a representative seed or averaged/selected.

7. The reference list should be checked for completeness and formatting. Several recent diffusion-channel and optimal-transport references are central to the paper and should be cited precisely.

---

## Overall Assessment

The manuscript presents a promising and communication-relevant adaptation of Sinkhorn/Wasserstein-flow-inspired drifting to conditional channel simulation. The condition-wise formulation is the main strength: it targets the correct object, namely \(p(y|x)\), rather than a global output distribution. The experimental results support the claim that condition-wise transport is preferable to joint/global transport within the drifting-family ablation.

However, the current paper is not yet ready for acceptance. The theoretical connection between the population flow and the implemented training algorithm needs sharper qualification, and the experimental comparison needs better controls or more conservative claims. The SSPA budget sensitivity is a particular concern and should be analyzed more thoroughly.

I therefore recommend **major revision**.

---

## Recommendation to the Editor

**Major Revision.** The paper has a publishable core idea and addresses a relevant problem for learned communication systems, but it requires significant clarification and additional experimental evidence before it meets the standard expected for IEEE Transactions on Communications.

---

## Acceptance Probability Estimate

- **Acceptance in current form:** 20–30%.
- **Eventual acceptance after strong revision:** 45–60%.

The paper has a defensible TCOM-level idea if positioned as a **conditional transport geometry plus low-latency channel simulator** paper rather than as a method that beats diffusion. The key revisions are:

1. tighten the theoretical claims;
2. add budget and hyperparameter sensitivity studies;
3. reduce comparative claims to the matched drifting-family comparison;
4. clarify downstream autoencoder evaluation;
5. emphasize the simulator-style repeated-conditional-sampling assumption.
