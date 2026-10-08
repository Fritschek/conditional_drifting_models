# From GANs to Diffusion and Drifting

This storyboard accompanies `slides.tex`. Slides 1--27 form the 20--25 minute main talk. Slides B1--B12 are backups.

## Main Talk

### 1. From GANs to Diffusion and Drifting

**Scientific purpose:** Frame the talk around generative channel simulation and the fidelity--latency tension.

**Content:** ITML/TUD title treatment, authors, Berlin 6G Conference, and the four recurring design anchors: fidelity, conditioning, differentiability, and fast sampling.

**Proposed visual:** A transmitted constellation passing through three generative routes: WGAN, iterative diffusion, and one-shot drifting.

**Equation:** None.

**Speaker narrative:** Wireless channel models are usually introduced through physical equations. In this talk, I will consider a complementary question: can a learned generator reproduce the conditional channel law well enough to act as a simulator? The comparison is organized around four requirements that matter in communication systems: fidelity, correct conditioning, differentiability, and sampling cost.

The path will lead from adversarial one-shot models to diffusion and then to drifting. Drifting is interesting here because it uses distributional dynamics during training while retaining a direct generator at inference.

**Transition:** We first need to define what a learned channel simulator is expected to reproduce.

### 2. A learned channel is a conditional sampler

**Scientific purpose:** Establish the exact target of channel learning.

**Content:** Physical channel (Y\sim p_{\rm ch}(y\mid x)), learned surrogate (\hat Y=G_\theta(X,Z)), and the conditional matching objective.

**Proposed visual:** Side-by-side physical channel and learned generator, both receiving the same (x).

**Equation:** (p_{G_\theta(X,Z)\mid X}(\cdot\mid x)\approx p_{\rm ch}(\cdot\mid x)).

**Speaker narrative:** The object we want to learn is not simply a received-signal distribution. A channel takes a transmitted input and returns a random output. The learned model therefore receives both the transmitted signal and fresh latent noise.

This viewpoint covers analytic simulators, channels with expensive physical simulation, and channels inferred from data. It also makes the surrogate differentiable with respect to its input, which is useful when it sits inside an end-to-end learned communication system.

**Transition:** That conditional sampler has to satisfy several requirements at the same time.

### 3. A useful simulator must satisfy four constraints

**Scientific purpose:** Introduce the recurring evaluation framework.

**Content:** High fidelity, correct conditioning, differentiability, and fast repeated sampling, with one concrete requirement stated for each.

**Proposed visual:** A compact four-quadrant diagram around a learned-channel block, followed by four equally weighted explanatory points.

**Equation:** None.

**Speaker narrative:** A model can generate plausible received samples and still be a poor channel simulator. It must associate each received distribution with the correct transmitted input. It should also expose gradients when used for communication-system learning, and it must remain affordable when called millions of times.

No single scalar metric captures all four requirements. This will matter later, when global sample-cloud scores and downstream error rates give different rankings.

**Transition:** WGAN channel models provide a natural first operating point on this design space.

### 4. GANs offer direct samples; WGAN is our baseline

**Scientific purpose:** Explain why GAN-based channel models are attractive, how WGAN refines the adversarial objective, and where the training difficulty lies.

**Content:** One generator call at inference; the original GAN framework; WGAN as the evaluated adversarial baseline; critic-based min--max training; possible instability and coverage issues.

**Proposed visual:** Two horizontal bands: direct one-shot inference on top and the generator--critic training loop below.

**Equation:** (z\xrightarrow{G_\theta(x,\cdot)}y).

**Speaker narrative:** GAN-based channel models have the inference path we would like: draw noise, combine it with the transmitted signal, and evaluate the generator once. WGAN replaces the original discriminator objective with a Wasserstein critic and is the adversarial baseline used in our experiments.

The compromise is training. The generator and discriminator or critic solve an adversarial problem, and the resulting behavior can depend strongly on optimization balance and model capacity. This motivated the move toward diffusion-based channel generators.

**Transition:** Diffusion changes the training problem and often improves distributional modeling.

### 5. GAN channel models established the differentiable-surrogate idea

**Scientific purpose:** Show that adversarial channel modeling developed into an active wireless research direction rather than remaining a generic machine-learning import.

**Content:** Conditional GAN channel surrogates for end-to-end learning, time-varying and frequency-selective models, MIMO channel generation, and digital-twin channel modeling.

**Proposed visual:** Three application cards progressing from conditional surrogates through channel memory to spatially structured and digital-twin models.

**Equation:** None.

**Speaker narrative:** GAN-based channel models established an important idea for learned communications: a generator can stand between a neural transmitter and receiver and provide both stochastic channel samples and a differentiable training path. Early conditional models focused on unknown channels in end-to-end learning, followed by frequency-selective and time-varying settings.

Later work extended the approach to MIMO impulse-response distributions, spatial statistics, and channel digital twins. This progression shows both the relevance of one-shot generators and the increasing burden placed on their conditional structure.

**Transition:** Diffusion models offer another route to learning these complicated channel laws.

### 6. Diffusion learns difficult channel laws through denoising

**Scientific purpose:** Give a visual, non-specialist explanation of diffusion channel models.

**Content:** Forward noising from channel samples to Gaussian noise; reverse iterative denoising conditioned on (x).

**Proposed visual:** Two horizontal chains, one for forward noising and one for reverse generation.

**Equation:** None in the main slide.

**Speaker narrative:** Diffusion training constructs a sequence of progressively noisier versions of the received sample. The learned reverse process then removes this noise step by step, conditioned on the transmitted input.

Prior channel-modeling work shows that this approach can represent AWGN, fading, and nonlinear channel distributions accurately without adversarial training. The price is paid when we generate a sample: the denoiser must be evaluated repeatedly.

**Transition:** Its impact on mainstream image generation explains why diffusion became the natural fidelity reference.

### 7. Diffusion reset the fidelity benchmark for image generation

**Scientific purpose:** Establish diffusion's broader generative-modeling importance before narrowing to wireless applications.

**Content:** The progression from DDPM to guided diffusion and latent diffusion; the associated gains in fidelity, coverage, and flexible conditioning; iterative sampling as the remaining cost.

**Proposed visual:** A three-step research timeline beside four reasons diffusion became central.

**Equation:** None.

**Speaker narrative:** Diffusion first became competitive for high-quality image synthesis, then surpassed leading GAN benchmarks while maintaining better distribution coverage. Latent diffusion subsequently made high-resolution and flexibly conditioned synthesis practical enough to underpin modern image generators.

This did not make every generative problem easy, but it substantially reduced the mode-coverage concern associated with adversarial training. The persistent tradeoff is iterative generation: every new sample still requires a sequence of denoiser evaluations.

**Transition:** Before returning to wireless communications, this familiar example makes the broader impact concrete.

### 8. Diffusion made text-conditioned generation tangible

**Scientific purpose:** Give the audience a brief visual pause after the conceptual diffusion discussion.

**Content:** The Stable Diffusion prompt ``a photograph of an astronaut riding a horse'' and its generated image.

**Proposed visual:** Full-height Wikimedia Commons image on a black background, with only the prompt and source attribution.

**Equation:** None.

**Speaker narrative:** This is the kind of example that made diffusion familiar far beyond machine learning: a short text prompt becomes a detailed image with coherent objects, composition, and style. The point here is not the astronaut or the horse. It is that diffusion became a practical model of complicated, conditioned distributions.

Wireless channel generation asks for the same broad capability, but with a conditioning variable that has a precise physical meaning.

**Transition:** That capability is now being adapted across the wireless stack.

### 9. Diffusion is now used across the wireless stack

**Scientific purpose:** Establish diffusion as an active wireless research direction and position the authors' channel-generation work within it.

**Content:** Channel-distribution generation and robust sampling, high-dimensional channel synthesis and digital twins, and diffusion priors for channel estimation and detection. The accepted IEEE TMLCN paper and the IEEE ICC paper by Kim--Fritschek--Schaefer are highlighted.

**Proposed visual:** Three application cards, with the authors' two channel-generation papers in an accent-colored card.

**Equation:** None.

**Speaker narrative:** Diffusion has quickly spread from generative channel simulation into high-dimensional user-specific channel synthesis, statistical channel digital twins, channel estimation, and joint estimation and detection. The breadth matters because all of these applications repeatedly invoke a learned stochastic model.

Our recently accepted IEEE TMLCN paper introduced conditional diffusion for accurate channel-distribution generation, and the subsequent IEEE ICC paper studied robustness and sampling tradeoffs in a communication setting. Those results motivate the present question: can we retain useful conditional fidelity with a direct sampler?

**Transition:** In a channel simulator, repeated denoising becomes a system-level cost.

### 10. Iterative sampling turns channel calls into system cost

**Scientific purpose:** Make inference cost concrete without overstating it.

**Content:** One-shot WGAN path versus (T)-step DDPM/DDIM path; distinction between model training and repeated channel use.

**Proposed visual:** One wide arrow for WGAN and a long sequence of small diffusion steps; an autoencoder loop beneath them.

**Equation:** (z_T\rightarrow z_{T-1}\rightarrow\cdots\rightarrow z_1\rightarrow y).

**Speaker narrative:** Diffusion training can be efficient, but each generated channel output still follows a sequential reverse trajectory. That distinction is important: training the channel model is usually done once, while sampling it may happen in every mini-batch of another optimization problem or throughout a large Monte Carlo study.

DDIM reduces the number of steps, but generation remains iterative. This leads to the question at the center of our work.

**Transition:** Can we retain direct generation without returning to adversarial optimization?

### 11. Can distributional training produce a one-shot generator?

**Scientific purpose:** Create the turning point of the talk.

**Content:** The research question in one sentence; desired training/inference split.

**Proposed visual:** Distributional training on the left and a single generator arrow on the right.

**Equation:** (z\xrightarrow{G_\theta(x,\cdot)}y).

**Speaker narrative:** We ask whether distributional dynamics can supervise a direct generator. The dynamics may be iterative during training, but they should be absorbed into the network parameters so that inference requires only one forward pass.

Drifting models provide exactly this separation. They construct targets by moving generated particles toward the data distribution and then train the generator to follow those targets.

**Transition:** The easiest way to understand drifting is to follow the particles.

### 12. Drifting learns a direct generator from particle directions

**Scientific purpose:** Explain the training principle intuitively before introducing the detached-target equation.

**Content:** Compare target and generated particles, construct attraction and repulsion directions, and train the generator toward the moved particles.

**Proposed visual:** Three-stage schematic: particle comparison, velocity construction, and network regression.

**Equation:** None.

**Speaker narrative:** We begin with two finite clouds: samples from the current generator and samples from the target channel distribution. A field assigns each generated particle a direction. Attraction pulls it toward the target cloud, while repulsion or a debiased self-interaction prevents the generated distribution from collapsing.

These moved particles are training targets for the generator. Repeating the update stores the transport in the network parameters. The field is therefore used during training, while inference remains one direct generator evaluation.

**Transition:** One detached target captures this training principle.

### 13. A detached drift target trains the direct map

**Scientific purpose:** Present the one main drifting equation used in the talk.

**Content:** Generate a particle, move it by a field, detach the target, regress the generator output toward it.

**Proposed visual:** A three-step pipeline with the equation centered underneath.

**Equation:** (\widetilde y=\operatorname{stopgrad}(\hat y+\eta v(\hat y)),\quad \min_\theta\|G_\theta(x,z)-\widetilde y\|^2).

**Speaker narrative:** The generator first produces a sample. A distributional field proposes a nearby target, and that target is detached from the computation graph. Ordinary regression then updates the generator toward it.

This is a practical training construction, not a claim that the finite neural-network update exactly follows a continuous gradient flow. The main design choice is therefore the field (v), especially once the data are conditional.

**Transition:** The conditional particle illustration shows how repeated updates realize this rule.

### 14. Repeated drift training reshapes conditional output clouds

**Scientific purpose:** Confirm the preceding explanation with the manuscript's conditional particle illustration.

**Content:** Initial particles, one drift update, repeated updates, and the distinction between iterative training and one-shot inference.

**Proposed visual:** Reuse the manuscript's conditional drifting toy figure with short training and inference callouts.

**Equation:** None beyond the update shown inside the figure.

**Speaker narrative:** This example shows the same mechanism for two conditions. The generated particles begin away from their target clouds. A drift update gives each particle a local direction, and repeated network updates reshape the two conditional output distributions.

The figure also reinforces the distinction from diffusion sampling. These repeated operations train the network. Once training is complete, a new sample still comes from one evaluation of the learned map.

**Transition:** A recent Wasserstein-flow construction provides a more principled way to choose the drift direction.

### 15. W-Flow turns Sinkhorn couplings into particle directions

**Scientific purpose:** Introduce the unconditional W-Flow/Sinkhorn development before presenting the wireless conditional adaptation, with complete attribution.

**Content:** A three-step construction: couple one generated particle to the target, couple it to the generated law itself, and subtract the two barycentric directions to remove entropic self-drift.

**Proposed visual:** Three separate panels for the target barycenter, self barycenter, and debiased W-Flow velocity.

**Equation:** (V_\varepsilon(u)=T^\varepsilon_{q,p}(u)-T^\varepsilon_{q,q}(u)).

**Speaker narrative:** Han, Li, Guo, Xu, Ermon, and Candès recently proposed W-Flow, which replaces the heuristic particle field with a velocity derived from the Sinkhorn divergence. One entropic transport coupling pulls generated particles toward target barycenters. A generated-to-generated self coupling removes the entropic self-bias and supplies the corresponding repulsive term.

This changes the training field, not the deployed model. W-Flow still uses the detached regression construction and retains one generator evaluation at inference. Its original formulation is unconditional, which brings us to the channel-specific issue.

**Transition:** Before specializing this construction to channels, its general image-generation results show why the idea is worth pursuing.

### 16. W-Flow moves one-step generation toward diffusion-level fidelity

**Scientific purpose:** Show the headline evidence for Sinkhorn-based W-Flow in its original unconditional setting.

**Content:** Han et al.'s reported ImageNet (256\times256) result: one-step generation, FID 1.29, approximately (100\times) faster sampling than multi-step diffusion models at similar FID, and improved mode coverage and domain transfer.

**Proposed visual:** A one-step generator path above three large result callouts.

**Equation:** None.

**Speaker narrative:** In the original image-generation setting, Han and colleagues report an FID of 1.29 on ImageNet at (256\times256) resolution from a one-step generator. Against multi-step diffusion models at similar FID, they report approximately one hundred times faster sampling, together with improved mode coverage and domain transfer over the compared one-step methods.

These are results for unconditional image generation, not wireless channels. They nevertheless show that a Sinkhorn-derived training field can combine broad distributional coverage with direct inference.

**Transition:** A channel introduces an additional constraint: each generated output must remain attached to the transmitted input that conditioned it.

### 17. Can we apply W-Flow directly to a wireless channel?

**Scientific purpose:** Motivate conditional transport with a memorable counterexample.

**Content:** Correct and permuted associations between transmitted symbols and received clouds; identical global mixture in both cases.

**Proposed visual:** Two constellation panels: correct colored clouds and color-swapped clouds, with the same gray marginal outline.

**Equation:** (p_Y(y)=\int p(y\mid x)p_X(x)\,dx) is insufficient.

**Speaker narrative:** Suppose the transmitted constellation has several points. A generator can reproduce the overall received mixture while attaching each cloud to the wrong transmitted symbol. A global distribution metric may then look good even though the input--output law is unusable.

For channel simulation, the relevant object is the output cloud at each fixed input. We call this fixed-input cloud an output fiber. The transport field should move outputs within these fibers rather than moving the transmitted input marginal.

**Transition:** This conditional requirement leads to two stages of our work on wireless channel simulation.

### 18. We adapt drifting to the conditional channel law

**Scientific purpose:** Distinguish the GLOBECOM parameterizations from the journal's conditional-transport extension.

**Content:** Two related stages of the work. The accepted GLOBECOM 2026 paper compares direct-output and residual drifting. The arXiv/TCOM journal submission studies direct-output, joint Sinkhorn, and condition-wise Sinkhorn under a matched one-shot architecture.

**Proposed visual:** Two labeled rows: the GLOBECOM parameterization study and the journal Sinkhorn drift-field extension.

**Equation:** None.

**Speaker narrative:** The accepted GLOBECOM paper studies two ways to parameterize the generated channel sample. Direct-output drifting generates (y) itself, while residual drifting generates a distortion term before reconstructing the output. This isolates the effect of channel representation under the original kernel drift.

The journal extension keeps the direct one-shot architecture and focuses on the field. Joint Sinkhorn transports samples in the combined input--output representation. Condition-wise Sinkhorn instead solves an output-space transport problem separately for each fixed transmitted input.

**Transition:** That last construction aligns the training geometry with the definition of a channel.

### 19. Condition-wise Sinkhorn transports only the output law

**Scientific purpose:** Explain the main technical contribution at a broad-audience level.

**Content:** Repeated target and generated outputs for the same (x); entropic OT couplings; target and self terms; one-shot network update.

**Proposed visual:** Three fixed-(x) rows with arrows only inside each row and no cross-condition links.

**Equation:** (\mathcal S^{\rm cond}_\varepsilon(q_\theta,p)=\int \mathcal S_\varepsilon(q_{\theta,x},p_x)\,\mu(dx)).

**Speaker narrative:** For each transmitted anchor (x), we draw several analytic channel outputs and several generator outputs. Sinkhorn iterations give a barycentric attraction toward the target cloud, while a generated-to-generated self term removes entropic bias and supplies the repulsive component.

The objective averages the Sinkhorn divergence over fixed-input conditional laws. The exact population objective has the desired conditional equilibrium under standard assumptions. Our implementation estimates its field with finite same-condition batches and uses the detached regression update from the previous slide.

**Transition:** The fixed-input output clouds show why this construction matters in practice.

### 20. The need for condition-wise Sinkhorn transport

**Scientific purpose:** Provide direct experimental evidence that condition-wise transport preserves the fixed-input channel law better than joint transport.

**Content:** Three fixed SSPA anchors: analytic, joint Sinkhorn, condition-wise Sinkhorn.

**Proposed visual:** Reuse and crop the manuscript's fixed-condition SSPA figure.

**Equation:** None.

**Speaker narrative:** Each row fixes one transmitted SSPA symbol. Gray points are analytic channel outputs. Joint Sinkhorn can place its global output mass plausibly while displacing or collapsing the cloud associated with a particular input.

Condition-wise Sinkhorn keeps the generated cloud near the analytic cloud for all three displayed anchors. This is the central visual argument for evaluating conditional channel laws rather than only the aggregated received distribution.

**Transition:** We test this behavior across channels with noise, fading, nonlinearity, and memory.

### 21. Four channels probe noise, fading, nonlinearity, and memory

**Scientific purpose:** Make the experimental progression meaningful.

**Content:** AWGN, elementwise Rayleigh fading without channel-state input/equalization, Rapp SSPA, and compact TDL-D-inspired short-block circular convolution.

**Proposed visual:** Four equally sized channel icons with one-line equations/labels.

**Equation:** Small channel expressions only.

**Speaker narrative:** AWGN is the controlled sanity check. Rayleigh introduces multiplicative uncertainty, and the downstream encoder and decoder do not receive the fading realization or an explicit equalizer. SSPA introduces nonlinear hardware distortion using the Rapp model.

The TDL experiment adds memory through a compact four-complex-use circular-convolution model inspired by the 3GPP TDL-D profile. It is a structured short-block stress test rather than a claim of full OFDM-scale channel realism.

**Transition:** The anchor-conditioned metrics confirm the conditional advantage across all four channels.

### 22. Condition-wise transport improves every anchor-conditioned metric

**Scientific purpose:** Quantify the conditional advantage without a dense paper table.

**Content:** Joint versus condition-wise anchor SWD and Gaussian Wasserstein-2 for AWGN, Rayleigh, SSPA, and TDL.

**Proposed visual:** Four compact before-to-after rows with exact values.

**Equation:** None.

**Speaker narrative:** Repeated samples at fixed transmitted anchors let us measure conditional means, spreads, and sliced projections directly. Lower is better in both columns. Condition-wise Sinkhorn improves both anchor metrics on every evaluated channel.

The gain is modest on AWGN and much larger on Rayleigh, SSPA, and TDL. This result isolates the effect of the training field because the two Sinkhorn variants use the same one-shot generator architecture.

**Transition:** The decisive communication check is what happens when these surrogates train an encoder and decoder.

### 23. Downstream SER reveals both transfer and remaining gaps

**Scientific purpose:** Present the strongest communication-relevance result.

**Content:** Four-channel SER curves for analytic, condition-wise Sinkhorn, WGAN, and DDIM-100; emphasize AWGN transfer and harder-channel gaps.

**Proposed visual:** Reuse the manuscript's four-panel SER curve figure with two short callouts.

**Equation:** None.

**Speaker narrative:** For each learned implant, we train the same symbolic autoencoder through the surrogate and evaluate the final encoder and decoder on the analytic channel. On AWGN, condition-wise Sinkhorn follows the analytic curve closely. On the harder channels, the learned models separate more clearly.

Diffusion is the strongest learned reference on the final Rayleigh, SSPA, and TDL curves. Condition-wise Sinkhorn is the strongest drifting-family surrogate in the symbolic experiments, but it does not universally dominate WGAN or diffusion. This is the fidelity side of the tradeoff.

**Transition:** The cost side becomes visible when the channel implant is placed inside a fixed-time training loop.

### 24. One-shot implants use the wall clock for optimization

**Scientific purpose:** Connect inference latency to end-to-end system training.

**Content:** Equal 1800-second autoencoder training budget; one-shot learned implants complete roughly 577k--642k updates, DDIM-100 roughly 21k; learned-method winner varies by channel.

**Proposed visual:** A common 1800-second timeline with many one-shot update ticks and fewer diffusion ticks; four channel badges naming the lowest learned SER.

**Equation:** None.

**Speaker narrative:** Here every channel implant receives the same 1800-second downstream training budget. The one-shot learned implants permit about 577,000 to 642,000 optimizer updates, depending on the channel and model. DDIM-100 permits about 21,000 because every channel call contains 100 denoising steps.

The lowest learned SER under this fixed budget is condition-wise Sinkhorn on AWGN and SSPA, DDIM-100 on Rayleigh, and WGAN on TDL. The result is not a universal ranking; it shows that sampling architecture changes how much communication-system optimization fits into a wall-clock budget.

**Transition:** Direct timing measurements place these observations on a latency--fidelity design frontier.

### 25. The best operating point depends on how the simulator is used

**Scientific purpose:** Synthesize latency and fidelity without a misleading aggregate Pareto plot.

**Content:** Measured inference ranges on RTX 5060 Ti; separate columns for standalone fidelity priority and inner-loop throughput priority.

**Proposed visual:** Log-like horizontal latency ruler: one-shot near (0.1\,\mu s), DDIM-100 near (39\,\mu s), DDPM near (45\)--(47\,\mu s); decision matrix beneath it.

**Equation:** One-shot cost (=1) generator evaluation; diffusion cost (=T) denoiser evaluations.

**Speaker narrative:** On the reported local GPU setup, drifting and WGAN require roughly (0.1\) microseconds per sample. DDIM-100 requires about (39\) microseconds, while DDPM is around (45\) to (47\) microseconds. These measurements use the same inference batch and include conditioning overhead.

If final standalone fidelity dominates, diffusion can justify the sequential cost. If the simulator is called repeatedly inside training or Monte Carlo loops, a one-shot model may provide the more useful operating point. Condition-wise Sinkhorn aims to move that one-shot point toward better conditional fidelity.

**Transition:** The complete fidelity--latency picture also identifies where the current study must be extended.

### 26. Open problems: time variation, nonlinearity, and MIMO

**Scientific purpose:** Connect the completed empirical synthesis to concrete wireless channel-modeling problems that remain open.

**Content:** Time-varying channels with Doppler and nonstationarity; nonlinear hardware with memory and coupled impairments; higher-dimensional MIMO channels with spatial correlation and geometry.

**Proposed visual:** Three compact schematics: a time-varying channel coefficient, a nonlinear block with state feedback, and a MIMO array linked through a channel matrix.

**Equation:** The common goal is to preserve (p(y\mid x)) as dimension, memory, and channel state grow.

**Speaker narrative:** The current channels isolate several important effects, but practical 6G surrogates must combine them. Time variation introduces Doppler, nonstationarity, and long temporal dependence. Hardware nonlinearities can themselves have memory and interact with other impairments. MIMO adds high-dimensional spatial correlation, array geometry, and environment state.

The open problem is therefore broader than scaling the network. The transport construction must preserve the conditional channel law while its dimension and state evolve.

**Transition:** These open problems lead to the broader design principle that closes the talk.

### 27. Design channel generators for fidelity, structure, and cost together

**Scientific purpose:** Close with three exact takeaways and forward-looking implications.

**Content:** Three takeaways; applications and limitations in one footer line.

**Proposed visual:** Three large numbered statements linked by a triangle of fidelity, structure, and efficiency.

**Equation:** None.

**Speaker narrative:** First, generative models provide flexible conditional simulators for channels that are complicated, mismatched, or data-driven. Second, diffusion offers strong distributional modeling, but sequential sampling can be expensive when the simulator is used repeatedly.

Third, conditional drifting explores a complementary point: direct generation without adversarial training. The current results also define the limitations clearly. Conditioning and representation design matter, high-dimensional measured channels remain open, and no single global sample metric replaces downstream communication evaluation.

**Transition:** I will stop here; the backup slides contain the transport details, complete numerical tables, and sensitivity checks.

## Backup Slides

### B1. Generic drifting uses attraction and repulsion

**Purpose/content:** Give the manuscript's row-normalized RBF attraction--repulsion field and detached regression update.

**Speaker narrative:** Plain drifting constructs a nonparametric velocity from target attraction and generated-sample repulsion. The generator is then regressed toward the detached moved particles. This baseline already has direct inference, but its interaction geometry does not enforce a fixed-input conditional law.

### B2. Optimal transport matches probability mass globally

**Purpose/content:** Define empirical probability measures, admissible couplings, marginal constraints, and the minimum-cost transport problem.

**Speaker narrative:** A coupling is a nonnegative matrix whose row and column sums reproduce the two probability laws. Each entry says how much mass is moved from one generated sample to one target sample. Optimal transport minimizes the total movement cost over all such globally mass-conserving couplings. With squared Euclidean cost, this gives the squared Wasserstein-2 distance.

### B3. Entropic OT produces the W-Flow velocity

**Purpose/content:** Define entropic OT, Sinkhorn divergence, the barycentric projection, and the target-minus-self W-Flow velocity.

**Speaker narrative:** Entropic regularization makes the finite transport problem efficiently solvable by Sinkhorn scaling. Each row of the optimal coupling defines a weighted target barycenter for its generated particle. Subtracting the generated self-barycenter removes entropic self-drift and yields the W-Flow velocity. We use ten Sinkhorn iterations and a minimum regularization of (10^{-3}).

### B4. The conditional population objective has the desired equilibrium

**Purpose/content:** State disintegration, conditional objective, and the qualified equilibrium result.

**Speaker narrative:** If the input marginal is shared, the joint laws disintegrate into fixed-input output laws. Nonnegativity and identity of indiscernibles for the Sinkhorn divergence then imply that a zero conditional objective means equality almost everywhere in (x). This is the population statement; finite-batch detached regression is an approximation to its field.

### B5. The finite-sample estimator uses repeated outputs per input

**Purpose/content:** Show generated, positive, and generated-reference samples per anchor and the field complexity.

**Speaker narrative:** The simulator setting lets us request repeated channel outputs for each fixed anchor. The implementation uses four generated, four target, and four independent generated-reference outputs per anchor. Condition-wise couplings scale linearly in the number of anchors, whereas expanded joint couplings carry a quadratic anchor factor.

### B6. Condition-wise Sinkhorn training update

**Purpose/content:** Reproduce the paper's six-step practical algorithm from condition-matched sampling through detached generator regression.

**Speaker narrative:** For every transmitted-input anchor, we draw generated particles, true channel outputs, and an independent generated reference cloud. Two Sinkhorn problems produce the target and self barycentric maps. Their difference moves each generated particle, and ordinary regression projects this explicit update back into the generator. These steps are repeated only during training; inference remains one direct network evaluation.

### B7. Direct-output SWD gives a different ranking

**Purpose/content:** Show the full direct-SWD table and seed/protocol caveat.

**Speaker narrative:** DDIM-100 has the lowest reference SWD on AWGN, Rayleigh, and SSPA. Within the drifting family, condition-wise Sinkhorn is best on AWGN, Rayleigh, and TDL, while direct drifting is best on SSPA. These global values are useful for comparability but do not reproduce the downstream ranking.

### B8. SSPA requires an update-budget-controlled operating point

**Purpose/content:** Show the four-row SSPA budget-sensitivity table.

**Speaker narrative:** The sharp same-condition SSPA field converges early. Extending it to the long diffusion-style preset causes severe degradation under fixed drift scale and small same-condition sample sets. The reported 30-seed model is the 4.69k-update operating point selected by anchor-conditioned metrics; the full preset is retained only as sensitivity evidence.

### B9. The AWGN result transfers to block length 64

**Purpose/content:** Show TurboAE BER/BLER curves and the exact 4 dB result.

**Speaker narrative:** We also tested the AWGN implant in a block-length-64, rate-one-half TurboAE-style setting. At 4 dB, analytic-channel training gives BER (1.96\times10^{-3}) and condition-wise Sinkhorn gives (3.30\times10^{-3}). The surrogate preserves the waterfall shape with a measurable gap.

### B10. Scope, reproducibility, and measured-data limitations

**Purpose/content:** Summarize model sizes, training protocol, hardware, repeated-condition assumption, and unvalidated local measured-data approximation.

**Speaker narrative:** The main W-Flow variants share the same one-shot generator, so their ablation isolates the drift field. Cross-family WGAN and diffusion comparisons use their stable reference protocols and are contextual rather than fully matched-capacity comparisons. Exact condition-wise sampling requires repeated outputs at the same input; the proposed local neighborhood approximation for sparse measured data remains future work.

### B11. Inference-network parameter counts

**Purpose/content:** Report the exact inference-network sizes and clarify that Sinkhorn couplings add training computation but no deployed parameters.

**Speaker narrative:** Direct and Sinkhorn drifting share a roughly 20k-parameter one-shot generator in all four channels. The diffusion denoiser and WGAN generator sizes vary by channel. This table documents capacity, but the cross-family experiments follow stable family-specific protocols rather than a fully matched-capacity design.

### B12. Selected references behind the method progression

**Purpose/content:** Give the principal diffusion, channel-diffusion, drifting, W-Flow, and Sinkhorn references used in the talk.

**Speaker narrative:** These are the main references behind the progression in the talk. The channel-diffusion papers establish the wireless baseline. The drifting and W-Flow papers supply the one-step training principles, while the Sinkhorn references provide the entropic transport construction used in our conditional adaptation.
