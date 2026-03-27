# Reviewer-Response Checklist

Working checklist for tightening the GLOBECOM conference version after external review feedback.

## 1. Novelty Framing
- Recast the paper as an experimental study / benchmark, not a major new communications framework.
- Keep the conditioning-aware kernel as an enhanced variant or case-study extension, not the headline benchmark claim.

## 2. Abstract Softening
- Reduce overclaiming.
- Make clear that the downstream kernel result is a case study, not the main table result.

## 3. Introduction Positioning
- Emphasize low-latency motivation, benchmark scope, and adaptation value.
- Mention mode-collapse only as motivation, not as a demonstrated result.

## 4. Contributions List
- Keep contributions factual.
- Avoid implying that the enhanced kernel already improves the benchmark table.

## 5. Methodology: Drifting Objective Interpretation
- Add a short paragraph connecting the drift field to kernel particle interactions / MMD-style intuition.

## 6. Methodology: Conditioning-Aware Kernel
- Rewrite the paragraph so it matches the good variant more closely:
  - condition-aware product kernel
  - target kernel may include residual similarity
  - not just a generic joint-kernel statement

## 7. Experimental Fairness Paragraph
- Add a short limitations/fairness paragraph:
  - practical benchmark, not matched-capacity
  - future matched-parameter / matched-FLOP / matched-budget study

## 8. Channel-Scope Paragraph
- State explicitly that the channels are low-dimensional / synthetic by design to isolate generator behavior before MIMO / memory settings.

## 9. Main Benchmark Results Paragraph
- Keep the story:
  - diffusion best fidelity
  - direct drifting strongest one-shot generator under direct-space SWD
  - residual drifting strong only in residual space
- Avoid tying the enhanced kernel to the main benchmark table.

## 10. Residual-Space Interpretation
- State explicitly that residual drifting is diagnostically useful but limited as a full channel surrogate in output space.

## 11. AWGN SER Figure Paragraph
- Make it a central caveat/result:
  - SWD does not map one-to-one to downstream SER
  - kernel design can matter downstream

## 12. AWGN SER Figure Caption
- Keep it honest and aligned with the text.

## 13. Metric-Limitations Remark
- Keep it short but central:
  - output-space SWD is not task-sufficient
  - joint / task-aware metrics are future work

## 14. Timing Methodology Sentence
- Add hardware / batching / warm-up / identical conditions in one concise sentence.

## 15. Timing Results Wording
- Keep the latency claim because it is one of the core paper drivers.
- Avoid sounding hand-wavy.

## 16. Conclusion
- Keep the conclusion modest:
  - benchmark / adaptation result
  - fidelity / latency tradeoff
  - direct vs residual distinction
  - need for downstream / task-aware evaluation
  - channel-specific kernel study left to future work

## 17. Optional Space Trim
- Shorten the metric-limitations remark or one introductory sentence if one reference spills to the next page.

## Recommended Order
1. Novelty framing
2. Abstract
3. Contributions
4. Methodology paragraphs
5. Fairness / scope paragraph
6. Results + AWGN figure text
7. Timing sentence
8. Conclusion
9. Space trimming
