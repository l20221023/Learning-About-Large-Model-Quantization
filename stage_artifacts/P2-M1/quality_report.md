# BF16 quality subset report

These are fixed 256-sample subsets, not complete official benchmark results.
Multiple choice uses generation-based option-letter scoring; no log-likelihood scores are mixed.
GSM8K extracts the final number after `####`, an explicit final-answer marker, or a numeric response ending. Ambiguous responses remain parse failures.
The independent audit re-tokenized stored messages with the configured chat template; local input token counts matched API prompt token usage for every request. API input/output token counts are retained per row.
Frozen sample SHA256: 814fcf6fed16408a284ed478adce700795e774131d926c1ca3a286d6e064c027
Raw result SHA256: e1dbb7d95a4dee36da28b8942f65e605aff4752b375071b086187fba6a65bebb
Audit errors: 0

| Dataset | N | Correct | Wrong | Failures | Accuracy / 256 |
|---|---:|---:|---:|---:|---:|
| gsm8k | 256 | 31 | 225 | 0 | 0.1211 |
| ceval | 256 | 135 | 116 | 5 | 0.5273 |
| mmlu | 256 | 131 | 125 | 0 | 0.5117 |

## Audit details
