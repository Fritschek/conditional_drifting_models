# Local ofdm development

Profile: pilot; seed: 9401.

Development outcomes only. Final confirmation and model selection remain separate.

## Generator cost

| Family | Parameters | Oracle outputs (including teacher) | Training seconds | Mean input energy per anchor |
|---|---:|---:|---:|---:|
| gaussian | 600611 | 2560000 | 42.48 | 63.99 |
| mixture | 598034 | 2560000 | 61.88 | 63.99 |
| sinkhorn | 599683 | 2560000 | 42.20 | 63.99 |
| flow | 599651 | 2560000 | 19.79 | 63.99 |
| consistency_distilled | 599651 | 2560000 | 24.91 | 64 |

Learned constellation: False. All receiver tests use the physical channel.

| Method | Es/N0 | BER | Block error rate | Inactive BER | Active BER | Receiver seconds |
|---|---:|---:|---:|---:|---:|---:|
| analytic | 0 | 0.268186 | 0.999756 | 0.268186 | n/a | 14.97 |
| analytic | 5 | 0.145041 | 0.958984 | 0.145041 | n/a | 14.97 |
| analytic | 10 | 0.063488 | 0.73407 | 0.063488 | n/a | 14.97 |
| analytic | 15 | 0.0288293 | 0.482666 | 0.0288293 | n/a | 14.97 |
| analytic | 20 | 0.0183928 | 0.353333 | 0.0183928 | n/a | 14.97 |
| gaussian_1 | 0 | 0.314322 | 1 | 0.314322 | n/a | 19.47 |
| gaussian_1 | 5 | 0.218928 | 1 | 0.218928 | n/a | 19.47 |
| gaussian_1 | 10 | 0.154606 | 0.999756 | 0.154606 | n/a | 19.47 |
| gaussian_1 | 15 | 0.123371 | 0.999084 | 0.123371 | n/a | 19.47 |
| gaussian_1 | 20 | 0.11159 | 0.998779 | 0.11159 | n/a | 19.47 |
| mixture_1 | 0 | 0.306641 | 1 | 0.306641 | n/a | 25.78 |
| mixture_1 | 5 | 0.203543 | 0.999878 | 0.203543 | n/a | 25.78 |
| mixture_1 | 10 | 0.135908 | 0.9953 | 0.135908 | n/a | 25.78 |
| mixture_1 | 15 | 0.103975 | 0.983826 | 0.103975 | n/a | 25.78 |
| mixture_1 | 20 | 0.0921718 | 0.973206 | 0.0921718 | n/a | 25.78 |
| sinkhorn_1 | 0 | 0.268045 | 0.999756 | 0.268045 | n/a | 14.18 |
| sinkhorn_1 | 5 | 0.16861 | 0.966309 | 0.16861 | n/a | 14.18 |
| sinkhorn_1 | 10 | 0.108434 | 0.823486 | 0.108434 | n/a | 14.18 |
| sinkhorn_1 | 15 | 0.083119 | 0.684814 | 0.083119 | n/a | 14.18 |
| sinkhorn_1 | 20 | 0.0739332 | 0.611633 | 0.0739332 | n/a | 14.18 |
| flow_32 | 0 | 0.266459 | 1 | 0.266459 | n/a | 52.23 |
| flow_32 | 5 | 0.165916 | 0.973206 | 0.165916 | n/a | 52.23 |
| flow_32 | 10 | 0.104653 | 0.837585 | 0.104653 | n/a | 52.23 |
| flow_32 | 15 | 0.0781206 | 0.691101 | 0.0781206 | n/a | 52.23 |
| flow_32 | 20 | 0.068646 | 0.615601 | 0.068646 | n/a | 52.23 |
| consistency_distilled_1 | 0 | 0.277443 | 0.999878 | 0.277443 | n/a | 15.41 |
| consistency_distilled_1 | 5 | 0.176397 | 0.97876 | 0.176397 | n/a | 15.41 |
| consistency_distilled_1 | 10 | 0.112643 | 0.86554 | 0.112643 | n/a | 15.41 |
| consistency_distilled_1 | 15 | 0.0840122 | 0.737183 | 0.0840122 | n/a | 15.41 |
| consistency_distilled_1 | 20 | 0.0734711 | 0.666931 | 0.0734711 | n/a | 15.41 |
| pilot_LMMSE_fixed_QPSK | 0 | 0.253823 | 0.99884 | 0.253823 | n/a | 0.00 |
| pilot_LMMSE_fixed_QPSK | 5 | 0.141857 | 0.948853 | 0.141857 | n/a | 0.00 |
| pilot_LMMSE_fixed_QPSK | 10 | 0.0641387 | 0.734802 | 0.0641387 | n/a | 0.00 |
| pilot_LMMSE_fixed_QPSK | 15 | 0.0250658 | 0.470642 | 0.0250658 | n/a | 0.00 |
| pilot_LMMSE_fixed_QPSK | 20 | 0.00931331 | 0.279053 | 0.00931331 | n/a | 0.00 |

| Method | Conditional SWD/floor | Pilot/data covariance error/floor |
|---|---:|---:|
| analytic | 0.8282 | 0.9589 |
| gaussian_1 | 2.007 | 4.541 |
| mixture_1 | 2.22 | 5.089 |
| sinkhorn_1 | 1.524 | 1.367 |
| flow_32 | 2.154 | 3.865 |
| consistency_distilled_1 | 2.19 | 3.869 |

The conventional receiver knows second-order channel statistics but not the hidden
realization or activity. Its rows use fixed QPSK even in learned-constellation runs.
BER uncertainty is calculated across independent blocks, not independent bits.
