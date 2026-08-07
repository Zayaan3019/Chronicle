# Loughran-McDonald word lists — provenance

`lm_negative.txt` (2,355 words) and `lm_uncertainty.txt` (297 words) are the
Negative and Uncertainty categories extracted from the Loughran-McDonald
Master Dictionary (2018 vintage), the standard finance/accounting-research
sentiment lexicon built specifically for analyzing SEC filing text:

> Loughran, T. and McDonald, B. (2011), "When Is a Liability Not a
> Liability? Textual Analysis, Dictionaries, and 10-Ks", *The Journal of
> Finance*, 66: 35-65.

Authoritative distribution point: https://sraf.nd.edu/loughranmcdonald-master-dictionary/

The full master CSV (86,553 words x per-category year-added flags) was
retrieved on 2026-08-07 from a public mirror of that same file —
`https://raw.githubusercontent.com/james-pavlicek/algorithmic-trading-with-artificial-intelligence/main/Loughran-McDonald_MasterDictionary.csv`
— and a word was kept for a list here if its `Negative`/`Uncertainty` column
was non-zero (the dictionary encodes "year added to this category" rather
than a boolean; 0 = not in the category). The extracted counts (2,355 / 297)
match the officially published category sizes for this dictionary vintage,
which is how this was verified as the genuine list rather than a
mis-labeled or truncated file.

`tone_signal.py` loads these two files as flat word sets; it does not use
any other column (word frequency, complexity, etc.) from the source CSV.
