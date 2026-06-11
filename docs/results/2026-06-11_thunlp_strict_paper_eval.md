# THUNLP-OPD strict paper eval results

Date: 2026-06-11

This records the completed THUNLP-OPD reproduction eval used as the comparison
baseline for later YOPD/RPI-OPD runs.

## Setting

- Eval script: `scripts/repro/run_thunlp_strict_paper_eval_all_ckpts.sh`
- Raw summary: `runs/thunlp_strict_paper_eval_20260610/summary.tsv`
- Data: `baselines/thunlp-opd/scripts/val/data`
- Tasks: `AIME24`, `AIME25`, `AMC23`
- Rollouts per question: `n=16`
- Sampling: `temperature=0.7`, `top_p=0.95`
- Max generation tokens: `31744`
- Thinking mode: `enable_thinking=false`
- Overall score: question-weighted mean over 30 AIME24, 30 AIME25, and 40 AMC23 questions.
- Teacher model path used in this run: `/mmu_mllm_hdd/zhangchenghao05/models/JustRL-DeepSeek-1.5B`
- Base model path used in this run: `/mmu_mllm_hdd/zhangchenghao05/models/DeepSeek-R1-Distill-Qwen-1.5B`

## Summary

| model | weighted | AIME24 | AIME25 | AMC23 | format errors |
|---|---:|---:|---:|---:|---:|
| base_orig | 0.4244 | 0.2479 | 0.2292 | 0.7031 | 142 |
| teacher_justrl | 0.6012 | 0.4854 | 0.3563 | 0.8719 | 113 |
| global_step_20 | 0.4644 | 0.3458 | 0.2354 | 0.7250 | 140 |
| global_step_40 | 0.5000 | 0.3167 | 0.2917 | 0.7937 | 108 |
| global_step_60 | 0.4981 | 0.3312 | 0.2625 | 0.8000 | 48 |
| global_step_80 | 0.5363 | 0.3771 | 0.3208 | 0.8172 | 58 |
| global_step_100 | 0.5131 | 0.3917 | 0.2458 | 0.8047 | 58 |
| global_step_120 | 0.5475 | 0.3771 | 0.3042 | 0.8578 | 76 |
| global_step_140 | 0.5550 | 0.4083 | 0.3104 | 0.8484 | 60 |
| global_step_160 | 0.5550 | 0.4333 | 0.3250 | 0.8187 | 68 |
| global_step_180 | 0.5631 | 0.4167 | 0.3333 | 0.8453 | 52 |
| global_step_200 | 0.5663 | 0.4167 | 0.3125 | 0.8688 | 77 |
| global_step_220 | 0.5944 | 0.4750 | 0.3792 | 0.8453 | 57 |
| global_step_240 | 0.5863 | 0.4729 | 0.3354 | 0.8594 | 85 |
| global_step_260 | 0.5613 | 0.4458 | 0.3104 | 0.8359 | 84 |
| global_step_279 | 0.5650 | 0.4083 | 0.3417 | 0.8500 | 98 |

## Takeaways

- Best THUNLP-OPD checkpoint: `global_step_220`, weighted score `0.594375`.
- Base score: `0.424375`.
- Teacher score: `0.601250`.
- Best checkpoint recovers `(0.594375 - 0.424375) / (0.601250 - 0.424375) = 96.1%` of the teacher-base gap.
- Final checkpoint `global_step_279` regresses to `0.565000`; for comparison, use `global_step_220` as the best THUNLP-OPD reproduction point and `global_step_279` as the terminal checkpoint point.

## Comparison Target

For a YOPD/RPI-OPD comparison under the same paper setting, report at minimum:

- `weighted`
- `AIME24`
- `AIME25`
- `AMC23`
- `format_error_rollouts`
- eval output directory
- checkpoint identifier
- exact sampling setting
