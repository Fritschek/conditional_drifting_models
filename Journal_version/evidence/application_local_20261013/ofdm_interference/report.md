# Local ofdm development

Profile: pilot; seed: 9401.

Development outcomes only. Final confirmation and model selection remain separate.

## Generator cost

| Family | Parameters | Oracle outputs (including teacher) | Training seconds | Mean input energy per anchor |
|---|---:|---:|---:|---:|
| gaussian | 600611 | 2560000 | 40.20 | 63.99 |
| mixture | 598034 | 2560000 | 60.78 | 63.99 |
| sinkhorn | 599683 | 2560000 | 42.85 | 63.99 |
| flow | 599651 | 2560000 | 17.41 | 63.99 |
| consistency_distilled | 599651 | 2560000 | 24.72 | 64 |

Learned constellation: False. All receiver tests use the physical channel.

| Method | Es/N0 | BER | Block error rate | Inactive BER | Active BER | Receiver seconds |
|---|---:|---:|---:|---:|---:|---:|
| analytic | 0 | 0.269951 | 0.999451 | 0.26259 | 0.335374 | 14.13 |
| analytic | 5 | 0.158943 | 0.958374 | 0.143924 | 0.292428 | 14.13 |
| analytic | 10 | 0.0847822 | 0.76062 | 0.0635276 | 0.273687 | 14.13 |
| analytic | 15 | 0.0527633 | 0.53479 | 0.0286615 | 0.266973 | 14.13 |
| analytic | 20 | 0.0431355 | 0.416565 | 0.018153 | 0.265174 | 14.13 |
| gaussian_1 | 0 | 0.305945 | 1 | 0.299332 | 0.364719 | 19.26 |
| gaussian_1 | 5 | 0.212096 | 0.999451 | 0.199296 | 0.325858 | 19.26 |
| gaussian_1 | 10 | 0.150338 | 0.994141 | 0.1326 | 0.30799 | 19.26 |
| gaussian_1 | 15 | 0.121796 | 0.986938 | 0.101439 | 0.302725 | 19.26 |
| gaussian_1 | 20 | 0.111083 | 0.983826 | 0.0896877 | 0.301238 | 19.26 |
| mixture_1 | 0 | 0.412222 | 1 | 0.40613 | 0.466366 | 25.23 |
| mixture_1 | 5 | 0.294558 | 1 | 0.278885 | 0.433857 | 25.23 |
| mixture_1 | 10 | 0.22239 | 0.999878 | 0.200702 | 0.415154 | 25.23 |
| mixture_1 | 15 | 0.191651 | 0.998108 | 0.167233 | 0.408677 | 25.23 |
| mixture_1 | 20 | 0.180697 | 0.995483 | 0.155239 | 0.406959 | 25.23 |
| sinkhorn_1 | 0 | 0.276265 | 0.999756 | 0.269375 | 0.337502 | 13.74 |
| sinkhorn_1 | 5 | 0.183607 | 0.96991 | 0.170825 | 0.297208 | 13.74 |
| sinkhorn_1 | 10 | 0.128254 | 0.846802 | 0.111205 | 0.279782 | 13.74 |
| sinkhorn_1 | 15 | 0.104877 | 0.725708 | 0.0859542 | 0.273062 | 13.74 |
| sinkhorn_1 | 20 | 0.0965696 | 0.664124 | 0.0769147 | 0.271257 | 13.74 |
| flow_32 | 0 | 0.275587 | 1 | 0.268404 | 0.339431 | 50.53 |
| flow_32 | 5 | 0.179986 | 0.981201 | 0.166812 | 0.297068 | 50.53 |
| flow_32 | 10 | 0.122319 | 0.868042 | 0.1047 | 0.278909 | 50.53 |
| flow_32 | 15 | 0.0976094 | 0.745544 | 0.0779551 | 0.272292 | 50.53 |
| flow_32 | 20 | 0.0888018 | 0.677612 | 0.0683699 | 0.270395 | 50.53 |
| consistency_distilled_1 | 0 | 0.289616 | 0.999939 | 0.282716 | 0.350941 | 15.34 |
| consistency_distilled_1 | 5 | 0.193702 | 0.985657 | 0.18047 | 0.311298 | 15.34 |
| consistency_distilled_1 | 10 | 0.132313 | 0.893311 | 0.114108 | 0.294115 | 15.34 |
| consistency_distilled_1 | 15 | 0.104551 | 0.785522 | 0.0839493 | 0.287654 | 15.34 |
| consistency_distilled_1 | 20 | 0.0944339 | 0.726685 | 0.0729249 | 0.285601 | 15.34 |
| pilot_LMMSE_fixed_QPSK | 0 | 0.260646 | 0.998901 | 0.253552 | 0.323702 | 0.00 |
| pilot_LMMSE_fixed_QPSK | 5 | 0.156408 | 0.953613 | 0.141883 | 0.285499 | 0.00 |
| pilot_LMMSE_fixed_QPSK | 10 | 0.0862301 | 0.76416 | 0.0654616 | 0.270815 | 0.00 |
| pilot_LMMSE_fixed_QPSK | 15 | 0.0524902 | 0.535706 | 0.0283129 | 0.267372 | 0.00 |
| pilot_LMMSE_fixed_QPSK | 20 | 0.0397551 | 0.392334 | 0.0142316 | 0.266602 | 0.00 |

| Method | Conditional SWD/floor | Pilot/data covariance error/floor |
|---|---:|---:|
| analytic | 0.8466 | 0.9767 |
| gaussian_1 | 1.89 | 4.1 |
| mixture_1 | 1.963 | 4.674 |
| sinkhorn_1 | 1.504 | 1.288 |
| flow_32 | 2.092 | 3.974 |
| consistency_distilled_1 | 2.085 | 3.881 |

The conventional receiver knows second-order channel statistics but not the hidden
realization or activity. Its rows use fixed QPSK even in learned-constellation runs.
BER uncertainty is calculated across independent blocks, not independent bits.
