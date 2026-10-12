# Independent assessment: stable SSPA simulation and downstream input coverage

12 October 2026. Reviewed commits `cf201be` and `0fea91a`. This assessment
supersedes earlier proposed next experiments where the current status index
marks them complete. No training, cluster submission or manuscript rewrite was
performed during this assessment.

## 1. Decision

The results materially strengthen the revision, but support a more specific
argument than “the most accurate generator produces the best communication
system.” There are now two useful application findings:

1. **Conditional grouping repairs the tested long-horizon failure.** Reallocating
   a fixed number of outputs into K64/B256 conditional clouds, including in the
   old trainer, succeeds where K4/B4096 fails. This now reproduces from scratch
   on three unused development seeds.
2. **The inputs chosen by the encoder matter as much as the inputs used to
   measure generator fidelity.** A surrogate can fit ordinary training inputs
   accurately while giving the optimizer attractive but inaccurate responses
   elsewhere. Broader characterization inputs substantially repair the Gaussian
   example. Sinkhorn shows a tradeoff, not a uniform improvement.

The strongest next step is a frozen replication of the coverage intervention,
followed by a genuinely structured channel. Do not repeat completed solver
studies, reopen the negative RBF-selector branch, or tune the radial mixture
repeatedly against these SERs. The earlier provisional venue direction does
not need to change on these development data alone; this assessment does not
recheck current venue policies.

## 2. What was independently verified here

The [portable-record audit](evidence/application_coverage_independent_audit_20261012.json)
can be regenerated from the checkout with:

```bash
python3 Journal_version/figures_src/audit_application_coverage_20261012.py
```

It checks both archive checksums and all 30 archived files against their current
counterparts, 35 local numerical-source hashes, the result/manifest/post-hoc
provenance chain, 21 application and 45 coverage curve points, integer SER/BER
identities, all three inherited control curves, matched coverage budgets, saved
codebook normalization, per-message error sums and the Gaussian radial-tail
calculation. The 24 common-codec gradient resolution flags agree with their
declared thresholds. Timing means agree with the saved measurements.

**Scope:** these are independent checks of portable records and local source,
not rerun GPU experiments. Neither new compact archive contains model
checkpoints. The October 10–11 component, intervention, replay and fresh-start
raw bundles are absent from this checkout; their conclusions below rely on the
tracked reports and inspected implementation, not a new checkpoint audit.
Original execution-host test and checkpoint-audit claims remain attributed to
those reports. External upstream source copies were not independently rebuilt.

### A normalization clarification

`normalized_codebook` subtracts **one scalar mean across all 64 x 8 entries**,
then normalizes their mean square to one. It does not impose a zero vector mean
across messages. This matches the saved codebooks: their grand means are below
7e-8, whereas individual coordinate means reach magnitude 0.3173. Average
squared codeword norm remains eight in every row.

Consequently the valid individual bound is
\(\|x_m\|\le\sqrt{64\cdot8}=\sqrt{512}\). Scalar centering does not tighten
that bound in dimension at least two: concentrate all energy in one codeword
whose coordinates sum to zero, with the others zero. The smaller bound
\(\sqrt{(64-1)8}\) would require **vector** centering and is not applicable to
these runs. Distinct codewords can approach the same supremum. This corrects
an interpretation of “centering,” not the numerical experiment. A new
coordinate-wise centering convention would require a separately named protocol.

## 3. Stability is now a practical result, with a precise scope

The [fresh-start report](sspa_fresh_results_20261011.md) gives:

| Raw final generator, 390720 updates | Late gate passed | Independent conditional SWD | Conditional variance |
|---|---:|---:|---:|
| Old K4/B4096 | 0/3 seeds | mean 1.27721 | 9.6743–9.8646 |
| Old K64/B256 | 3/3 | mean 0.0208845 | 0.052574–0.052778 |
| Official normalized K64/B256 | 3/3 | mean 0.0200636 | 0.052744–0.052879 |

The matching evaluation floor is 0.0183507 and true variance is 0.0528298.
The old-K64 bridge and fresh runs retain the old solver, loss, fixed epsilon,
optimizer and clipping. Therefore explicit solver corrections and the complete
official recipe are unnecessary for this particular rescue.

Call the intervention **grouping at matched output exposure**. Increasing K
also decreases independent conditions B by 16x. At fixed N=BK, transport matrix
work scales as BK²=NK, so equal output counts do not mean equal transport
compute. Identical initial weights/RNG do not make later noise draws identical
after changing the number of input draws. These facts do not invalidate the
intervention, but prevent attributing it solely to lower gradient variance.

The solver-only branches exclude the tested numerical changes as sufficient
repairs. They do not establish failure of a fully converged OT solver under all
settings, or rule out numerical errors amplifying deterioration. Nor do three
finite trajectories prove asymptotic stability, minimal K, or other-channel
reliability. Empirical rank deficiency of one K4 cloud is not a proof that the
conditional law cannot be learned over repeated batches.

The full official package improves gradients at the six common codecs beyond
old K64. Its extra components remain confounded for that endpoint. Healthy
methods' final SER rankings are mixed; selected K4 is still competitive.
Thus stability, local gradient fidelity and final communication performance
are three separate results.

## 4. The coverage intervention: strong Gaussian evidence, mixed Sinkhorn evidence

Independent reconstruction of nominal-SNR symbol errors, each out of 1,048,576:

| Generator | AE seed | Normal-input errors | Broader-input errors | Normal SER / broader SER |
|---|---:|---:|---:|---:|
| Gaussian | 209111 | 3251 | 43 | 75.60 |
| Gaussian | 209112 | 871 | 100 | 8.71 |
| Gaussian | 209113 | 4745 | 117 | 40.56 |
| Sinkhorn | 209111 | 81 | 104 | 0.779 |
| Sinkhorn | 209112 | 36 | 42 | 0.857 |
| Sinkhorn | 209113 | 468 | 99 | 4.73 |

These are three downstream initializations of **one generator per arm**, not
three generator replicas. Gaussian improves in every pair at 4, 6 and 8 dB.
Sinkhorn worsens in the first two pairs at all three SNRs and improves the last
pair. A lower mean Sinkhorn SER is driven by repairing its worst control case.
The small nominal-SNR differences should not be called statistically resolved
without paired error indicators or an appropriate uncertainty calculation.
Millions of evaluation draws do not replace independent training runs.

The preceding post-hoc localization is unusually informative: two Gaussian
codewords have norms 10.27 and 10.05 and contribute 1532/1560 physical-channel
errors. The radial Gaussian probability beyond 10.27 is approximately 3.18e-19.
Even 40.96 million independent training conditions would have expected count
only about 1.30e-11 beyond that radius. This is a radial tail calculation, not
a certificate of local coverage in a particular direction.

The intervention changes only the generator input law within each family.
It then improves means at tail inputs, reduces the optimistic surrogate-to-real
gap, and produces less extreme codebooks without a new peak-power constraint.
Together these observations support inadequate training coverage as a
contributor to failure. They do not isolate a unique causal mediator: changed
fitted values, derivatives and subsequent optimization all participate.

The tradeoff is real. Sinkhorn's ordinary-input SWD/floor changes from 1.200 to
1.455; variance at radius sqrt(512) is still only 0.587 of truth. Gaussian's
ordinary-input ratio changes from 1.097 to 1.061 and its tail variance remains
much closer to truth. A mean-only extrapolation repair would therefore hide
remaining Sinkhorn distribution error.

Both coverage arms retain physical noise and AE average power. The broader
query law has expected squared norm 48.6667 instead of eight, **6.083x the query
energy**. This is a simulator-access characterization experiment, not an
equal-energy RF measurement experiment. The normal mixture component remains
unbounded. Use the original fixed mixture for replication; do not infer that
its 25% fraction or uniform-radius density is optimal.

## 5. What the fast comparison establishes

At the resolved early common codec, Gaussian and Sinkhorn encoder-gradient
cosines are about 0.991, yet they have very different final transfer outcomes.
WGAN's cosine is only 0.627, with norm ratio about 147, but its trained code has
the lowest numerical SER. These are direct obstacles to claiming a universal
pre-optimization scalar ordering from this local gradient check. All final
common-codec references in this pilot are unresolved and must remain so.

WGAN's poor law fidelity cannot be erased because its code works well. Its
own-codebook variance is approximately three times the physical variance;
surrogate SER is about 0.00289 while the physical panel records zero errors.
Training with a pessimistic channel may act as a robustness intervention, but
the present experiment does not isolate that explanation. At 8 dB the main
WGAN/analytic comparison is two versus six errors, insufficient evidence of a
training-method advantage from one run. At lower SNR its numerical advantage
is larger, but the generator/AE replication problem remains.

The speed claim also needs the correct denominator:

| Model | Generator training, seconds | One AE training, seconds | Sum |
|---|---:|---:|---:|
| Gaussian | 42.80 | 37.48 | 80.28 |
| WGAN | 68.32 | 36.21 | 104.53 |
| DDIM-20 | 51.13 | 362.86 | 414.00 |
| Flow-8 | 42.90 | 88.70 | 131.60 |
| Sinkhorn | 136.64 | 37.20 | 173.84 |
| Distilled consistency, teacher included | 81.59 | 41.77 | 123.36 |

Sinkhorn makes AE training about 9.75x faster than DDIM-20 in this implementation;
including one generator training gives about 2.38x. It is not the cheapest
complete pipeline against every fast baseline. These sums exclude development,
evaluation, initialization and I/O. Batch-one measurements and amortized
batch throughput remain separate; measured NFE implementations are not proven
per-family optima. The SSPA conditional law itself is Gaussian, so successful
Gaussian baselines are scientifically expected, not an inconvenient exception.

## 6. A precise theory direction: transfer over feasible inputs

The following are elementary proved bounds, not a novelty claim or a validated
new empirical score. They explain what is missing from average-input fidelity.

Let P_x and Q_x be probability kernels on Euclidean channel outputs. Let mu be
the continuous input law used for training/evaluation, and C=(x_1,...,x_M) an
allowed deterministic codebook. Its input law is
\(\nu_C=M^{-1}\sum_m\delta_{x_m}\). Since mu has no atoms,
\(\nu_C\not\ll\mu\): an ordinary finite density-ratio bound cannot transfer
an average under mu to this discrete law. This does not rule out bounds using
regularity; it identifies the missing assumption.

For example, take P_x=N(0,1) and
\(Q_x=N(a\psi((x-x_*)/h),1)\), where a is nonzero and psi is a smooth compactly
supported bump with psi(0)=1. As h tends to zero, integrated conditional TV (or
any finite-power Wasserstein discrepancy) under nonatomic mu tends to zero,
while the discrepancy at x_* stays nonzero. This follows from the shrinking
support, its vanishing mu mass and the fixed bounded mean shift. Smoothness
without a uniform regularity bound does not prevent the example.

### A coverage-to-uniform-error bound

Let S be a full-dimensional feasible input ball, e:S->[0,infinity) a measurable
L-Lipschitz error, 1<=p<infinity, and
\(\int_S e(x)^p\,\mu(dx)\le\varepsilon^p\). For x in S and r>0 with
\(b_x(r)=\mu(B(x,r)\cap S)>0\),

\[
e(x)\le Lr+\frac{\varepsilon}{b_x(r)^{1/p}}.
\]

**Proof.** On the ball, e(y)>=max(e(x)-Lr,0). Integrating its p-th power yields
\(b_x(r)(e(x)-Lr)_+^p\le\varepsilon^p\), giving the claim. Taking a supremum
over x gives the same bound with \(\inf_{x\in S}b_x(r)\), when positive.
Zero small-ball mass makes the bound vacuous. A radial tail probability supplies
no required lower bound on this local mass. The integral uses the original mu,
not a silently renormalized restriction to S.

For the present average-power constraint S can be the ball of radius sqrt(512).
Choosing only a sphere would be invalid with ambient Gaussian mu, since that
sphere has zero mu mass. The unknown L and difficult small-ball coverage in
eight dimensions are substantial obstacles to a numerical certificate.

### A decoder-independent risk-transfer certificate

Use \(\operatorname{TV}(P,Q)=\sup_A|P(A)-Q(A)|\). For a codebook C and any
measurable decoder, let R_P and R_Q denote the same message-averaged loss, with
loss values in [0,1]. Then

\[
|R_P(C,d)-R_Q(C,d)|
\le\frac1M\sum_m\operatorname{TV}(P_{x_m},Q_{x_m}).
\]

**Proof.** For each message, the difference of expectations of a [0,1]-valued
measurable function is at most TV, by integrating its level sets. Average these
inequalities and use the triangle inequality.

Define Delta as the supremum of the right-hand side over the allowed codebook
class. It is independent of a decoder and requires no downstream AE training
to define. The simpler uniform bound sup_{x in S} TV(P_x,Q_x) also bounds Delta.
If eta>=0 and theta_Q is an eta-approximate minimizer of **this same surrogate risk over
the same codebook/decoder class**, then

\[
R_P(\theta_Q)-\inf_\theta R_P(\theta)\le 2\Delta+\eta.
\]

**Proof.** Compare R_P(theta_Q) to R_Q(theta_Q), use approximate optimality,
then compare the surrogate risk of any competing theta to its physical risk.
Take the infimum over competitors; no attained minimizer is required.

This can bound SER transfer for a fixed codec without any gradient assumption.
It does **not** certify the current AE endpoints: training minimizes unbounded
cross-entropy, not SER, and the relevant optimization gap is unknown. The direct
Wasserstein bound for uniformly Lipschitz losses does not apply to SER indicators;
an extension would require additional assumptions, such as control of channel
mass near decoder decision boundaries. Empirical SWD or a finite moment bank
does not automatically upper-bound TV, and sampled radial checks do not certify
a supremum over all feasible codebooks.

This is a useful reformulation of the earlier metric ambition: a conservative
transfer certificate over a declared feasible-input/loss class is plausible;
an accurate universal ranking of final stochastic optimization outcomes is not
established. Input derivatives remain relevant to the optimization gap and
trajectory, but one frozen-codec cosine does not control them globally.

## 7. Concrete next handover

1. **Replicate the unchanged coverage intervention before enlarging it.** Use
   three newly allocated generator seeds, two families (Gaussian/Sinkhorn), two
   input laws and three paired AE initializations per generator seed. This is
   12 generators, 36 surrogate AE runs and nine paired analytic references.
   Retain 40k generator/20k AE updates, 40.96m positive outputs per generator,
   final EMA selection, the exact mixture and current normalization. Generator
   training alone totals 491.52m positive observations; evaluation is additional.
   Record generator-seed blocks separately from AE and Monte Carlo replication.
   The exact seed IDs must be allocated from unused IDs before execution.
   Extend the existing runner's hard-coded parent/seeds through a new protocol
   and child manifest; do not relabel 9111 or override its frozen parent.

2. **Retain the full panel, not just mean SER.** Report each paired SER change
   at 4/6/8 dB, surrogate optimism, maximum codeword norm, tail mean/variance,
   ordinary-input fidelity, output-query energy and measured cost. Save
   per-message errors and paired per-trial disagreements or equivalent paired
   sufficient statistics when testing small evaluation differences. Define a
   primary endpoint before running; do not select it from the new outcomes.
   If Gaussian replicates but Sinkhorn remains mixed, report that result and
   retain the coverage/accuracy tradeoff rather than tune away its failures.

3. **Before final family rankings, check the saved samplers.** A bounded
   fixed-checkpoint panel (for example Flow 8/32/128 and DDIM 20/50/100 NFEs)
   should check conditional variance, fidelity, input derivatives and measured
   cost on common inputs/latent draws. These are proposed values, not completed
   tests or a convergence guarantee. Record the downstream choice in advance;
   do not choose NFE by the final paper-test SER. Apply any chosen coverage law
   consistently across competitors, including distilled consistency, while
   preserving the Gaussian-input rows.

4. **Move the publication question to a structured channel.** After those
   bounded checks, carry the credible one-shot recipe and strong baselines to
   the planned joint wireless block with shared randomness/nontrivial joint
   structure. SSPA alone cannot demonstrate a need for a flexible non-Gaussian
   conditional generator. Keep the requested optical-fiber K4/K64 pilot as a
   separate bounded follow-up, not evidence already obtained. A WGAN-inspired
   noise-augmentation control is optional if its robustness mechanism becomes
   a paper claim; do not add it merely to chase the best pilot SER.

5. **Revise the manuscript around the evidence.** State the inherited
   conditional W-Flow mechanism and simulator access; explain grouping and
   its exposure/compute tradeoff; show the failure and repair; distinguish
   training-law fidelity, feasible-input fidelity, gradients and analytic-channel
   utility; report end-to-end costs with teacher training. Keep unproved
   stability mechanisms and general metric claims out of theorem/conclusion
   statements. The elementary bounds above need literature positioning before
   any originality claim. Existing broader reviewer requirements still apply.

No new experiment was launched by this handover. The completed round warrants
progress to these bounded checks, not a new open-ended SSPA search.
