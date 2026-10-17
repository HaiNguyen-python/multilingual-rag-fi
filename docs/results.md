# Results

Data: synthetic, illustrative corpus (18 Finnish pages written for this project) and a hand-written eval set (40 answerable + 8 unanswerable questions, each in FI, EN and VI); offline glossary translator and extractive answers.
Regenerate with `mlrag report`; every number below comes from that run.

## Retrieval and answers per question language

| strategy | lang | hit@3 | hit@5 | MRR | answer rate | key fact | citation valid | refusal acc. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| translate + BM25 | FI | 1.00 | 1.00 | 1.00 | 0.82 | 0.72 | 1.00 | 1.00 |
| translate + BM25 | EN | 0.97 | 0.97 | 0.98 | 0.72 | 0.62 | 1.00 | 0.75 |
| translate + BM25 | VI | 0.97 | 0.97 | 0.95 | 0.57 | 0.50 | 1.00 | 1.00 |
| char n-grams (baseline) | FI | 1.00 | 1.00 | 0.99 | 0.93 | 0.82 | 1.00 | 0.88 |
| char n-grams (baseline) | EN | 0.25 | 0.28 | 0.22 | 0.00 | 0.00 | n/a | 1.00 |
| char n-grams (baseline) | VI | 0.12 | 0.15 | 0.12 | 0.00 | 0.00 | n/a | 1.00 |
| hybrid (RRF) | FI | 1.00 | 1.00 | 1.00 | 0.75 | 0.68 | 1.00 | 1.00 |
| hybrid (RRF) | EN | 0.97 | 0.97 | 0.96 | 0.72 | 0.65 | 1.00 | 0.75 |
| hybrid (RRF) | VI | 0.97 | 1.00 | 0.93 | 0.53 | 0.45 | 1.00 | 1.00 |
| hybrid + rerank | FI | 1.00 | 1.00 | 1.00 | 0.82 | 0.70 | 1.00 | 1.00 |
| hybrid + rerank | EN | 0.97 | 0.97 | 0.97 | 0.72 | 0.62 | 1.00 | 0.75 |
| hybrid + rerank | VI | 0.95 | 1.00 | 0.95 | 0.57 | 0.47 | 1.00 | 1.00 |
| dense (multilingual-e5) | all | not run here: needs the `[neural]` extra and a model download | | | | | | |

## The cross-lingual gap (MRR)

Gap = MRR on the Finnish questions minus MRR on the same questions in English or Vietnamese. 'Closed' = share of the char n-grams (baseline) gap that the strategy removes.

| strategy | gap FI-EN | gap FI-VI | closed EN | closed VI |
| --- | --- | --- | --- | --- |
| translate + BM25 | 0.022 | 0.051 | 97% | 94% |
| char n-grams (baseline) | 0.772 | 0.867 | 0% | 0% |
| hybrid (RRF) | 0.037 | 0.066 | 95% | 92% |
| hybrid + rerank | 0.033 | 0.051 | 96% | 94% |

## Refusal thresholds

Chosen by a sweep over 0.00-1.00 (step 0.01) on all three languages pooled: balanced accuracy of answer rate (answerable) and refusal rate (unanswerable), middle of the widest best range. `hybrid` uses the shipped default (`DEFAULT_MIN_RELEVANCE`).

| strategy | relevance score | best range | threshold used |
| --- | --- | --- | --- |
| translate + BM25 | IDF-weighted term coverage | 0.41-0.41 | 0.410 |
| char n-grams (baseline) | cosine of best hit | 0.22-0.25 | 0.235 |
| hybrid (RRF) | IDF-weighted term coverage | 0.41-0.41 | 0.410 |
| hybrid + rerank | IDF-weighted term coverage | 0.41-0.41 | 0.410 |

## Glossary translator coverage

Share of content words in the answerable questions that the offline glossary translator recognised: EN 76%, VI 70%. Unknown words are kept as typed and lower the relevance score.

## Read with care

- Small eval set: 40 answerable questions per language, so one question moves a rate by 0.025. Differences of 0.01-0.03 between strategies are one or two questions.
- I wrote the pages, the questions and the glossary myself. The glossary was written from the corpus vocabulary and not tuned after the eval ran, but the same author bias makes these numbers optimistic compared with real newcomers' questions.
- The refusal threshold is chosen on the same questions it is scored on, and the best range is a single grid point for the coverage-based strategies, so it is fragile. Two code fixes (the stemmer's final-vowel trim and word-level sentence scoring) were made after reading eval misses.
- English and Vietnamese answers quote the Finnish source sentence with a glossary gloss; they are not translations. 'key fact' only checks that the quoted sentence holds the expected value.
- The n-gram baseline answers nothing in EN/VI: the extractive generator picks sentences that share words with the question, and an untranslated question shares none with Finnish text ('n/a' = no answers to check).
