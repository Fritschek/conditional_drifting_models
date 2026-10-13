# Ideal Gaussian flow: Euler discretization check

Derived before the full saved-sampler panel completed. This is an analytic
reference calculation, not a fitted model result or a new convergence theorem.

The pinned Meta `CondOTProbPath` uses `(1-t) Z + t Y`, and the adapter draws
Z~N(0,I) independently of Y~N(m(x),v I) at fixed channel input x. Its population
conditional mean velocity, obtained by Gaussian conditioning, is

```
s2(t) = (1-t)^2 + v*t^2
a(t) = ((1+v)*t - 1) / s2(t)
u(t,y;x) = m(x) + a(t)*(y - t*m(x)).
```

The mean follows `t*m(x)` exactly even under left-endpoint explicit Euler on a
uniform grid. The scalar noise coefficient instead satisfies

```
c_0 = 1
c_{i+1} = c_i * (1 + a(i/N)/N)
Var_Euler(Y_1 | x) / v = c_N^2 / v.
```

For the frozen nominal-8-dB SSPA real-coordinate variance
`v = 1/(4*0.75*10**0.8) = 0.052829773...`, direct evaluation gives:

| Euler evaluations | Ideal sampled variance / target variance |
|---:|---:|
| 8 | 0.55328954 |
| 32 | 0.86595650 |
| 128 | 0.96471523 |

Thus low conditional variance at Flow-8 can arise from discretization even if
the trained velocity is exact. The learned-model measurements should be compared
with these reference values, while retaining network approximation and finite-MC
error as separate possibilities. This calculation alone does not identify how
much of an observed neural-model discrepancy is numerical.

In this ideal constant-variance case the input Jacobian remains Dm(x), since
the Euler mean is exact and its noise coefficient is input independent.
Expected loss gradients can nevertheless change because the sampled conditional
variance changes. This distinction matters for interpreting a pathwise Jacobian
check versus the expected-loss gradient at a fixed decoder.

The calculation is specific to independent endpoint coupling and this Euler
sampler. It does not imply the same bias for other flow couplings, solvers,
distilled models, or non-Gaussian conditional channels. In particular, the
upstream path's name does not mean that the independent endpoints are already
coupled by the condition-wise optimal transport map.
