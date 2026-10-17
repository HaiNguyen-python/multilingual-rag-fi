# How it works

This page follows one question through the system: **"Vé tháng 30 ngày giá bao nhiêu?"**
(Vietnamese for "How much is a 30-day season ticket?"). File references point to `src/mlrag/`.

## 1. The corpus and the index

`mlrag index` -> `index.py: build_index`

1. `corpus.py: load_pages` reads every `corpus/*.md`. A file holds several pages; each starts
   with a marker comment `<!-- page: matkakortti | en: Travel card ... | vi: Thẻ đi lại ... -->`
   and a `# Title`. The English and Vietnamese titles let an answer name its source in the
   question's language. Other comments (the "illustrative, not official guidance" notice) are
   dropped.
2. Each page gets a **content hash** (SHA-256 of id, titles and body). If the hash equals the one
   in `index/index.json`, the page is kept as it is. New and changed pages are chunked again,
   and pages that disappeared are removed. The file is written to a temp file and renamed, so a
   crash never leaves half an index.
3. `chunk_page` makes **one chunk per `##` section**, ids like `matkakortti#1`. A citation then
   points to one topic ("Kausilippu"), not a whole page. Very long sections are split at
   sentence borders, never across a heading.

## 2. Finnish analysis

`text.py: FinnishAnalyzer`, built from all chunk texts when the retriever starts.

- **Normalise:** Unicode NFC (so "ä" typed as a + combining mark matches), lower case, one
  kind of dash, no soft hyphens. Tokens are words or numbers with a decimal comma ("3,20").
- **Stem:** the Snowball Finnish stemmer, then one extra step. Snowball does not undo
  consonant gradation, so forms of one word can end one vowel apart (Snowball gives vaippa ->
  `vaip` but vaipat -> `vaipa`). Dropping one final vowel from stems of 5+ letters joins them.
- **Split compounds:** "kirjastokortin" -> `kirjastokort` + `kirjast` + `kort` (whole stem
  plus the stems of both parts). A split is
  accepted only if both parts are stems already seen in the corpus, and only if the tail is a
  real word and not just a case ending ("terveysasema + lle" is not "terveysasema + alle").
  Stopwords and stems shorter than 3 letters never count as parts.

So the Finnish query word "kortti" matches the document word "kirjastokortin".

## 3. Language detection and translation

`text.py: detect_language` scores Vietnamese-only letters (ơ, ư, đ, tone marks), Finnish
ä/ö/å and short function words. Our question has many Vietnamese marks -> `vi`.

`translate.py: GlossaryTranslator.to_finnish` walks the tokens and, at each position, tries the
longest phrase in the glossary first (up to the longest key, e.g. 5 syllables):

| tokens | match | output |
| --- | --- | --- |
| vé tháng | `vé tháng: kausilippu` | kausilippu |
| 30 | number, kept | 30 |
| ngày | unknown, kept | ngày |
| giá bao nhiêu | `giá bao nhiêu: maksaa` | maksaa |

Result: `kausilippu 30 ngày maksaa`, coverage 3/4 (the number counts as understood). Phrase
lookup matters most for Vietnamese, where one word is several syllables ("thẻ thư viện").
Multi-syllable phrases also get a diacritic-free alias, because many people type Vietnamese
without marks. `LLMTranslator` does the same job with a chat model when a provider is set.

## 4. Retrieval strategies

`retrieval.py: Retriever.retrieve(question, lang, strategy)`

| strategy | what it searches with |
| --- | --- |
| `translate_bm25` | BM25 (`lexical.py`) over the analysed Finnish query |
| `ngram` | character 3-5-gram TF-IDF on the question **as typed** (baseline) |
| `hybrid` | RRF of BM25 and n-grams, both on the translated query |
| `hybrid_rerank` | hybrid, then the reranker over the top 20 |
| `dense` | multilingual-e5 on the raw question (`neural.py`, [neural] extra) |

**RRF** (`rrf`): `score = sum over lists of 1 / (60 + rank)`. It uses ranks only, so BM25 scores
and cosines never need to be put on one scale. The n-gram side helps when the stemmer misses a
form; BM25 helps when a short word would drown in n-grams.

**Relevance** (for refusal) is the IDF-weighted share of the query terms found in the top chunk
(`weighted_coverage`). For our question, `kausilip`, `30` and `maks` are in `matkakortti#1`,
`ngày` is not; since `ngày` never occurs in the corpus it has the highest IDF, so it costs a
lot: relevance 0.57. The n-gram baseline and dense retrieval use the cosine of the best hit.

**Lexical reranker:** re-sorts candidates by weighted coverage plus the fused score. The
cross-encoder (`neural.py`) does the same job with a multilingual model and the raw question.

## 5. Answer or refuse

`answer.py: Assistant.answer_from`

- If relevance < threshold (`DEFAULT_MIN_RELEVANCE = 0.41`), the answer is the refusal sentence
  in the question's language. No generator is called.
- `ExtractiveGenerator` scores every sentence of the top 4 chunks by how many query **words**
  it matches (not terms: "kirjastokortti" alone would count three times), then by chunk rank
  and position. It keeps the best two and cites each one.
  - Finnish question: the sentences with citations.
  - English / Vietnamese: a template in that language naming the page in both languages, the
    quoted Finnish sentences with citations, a gloss of up to 5 known words from the reversed
    glossary, and the line "this is a quote in Finnish, not a translation".
- `LLMGenerator` sends the Finnish chunks with ids and tells the model to answer in the
  question's language, cite ids in brackets, or reply with the exact refusal sentence.
- `validate_citations`: valid only if at least one chunk is cited and every cited id was in the
  context. An LLM that cites `[uimahalli#7]` for a bus question is caught.

## 6. Evaluation

`evals.py: evaluate`, run by `mlrag eval` and `mlrag report`.

1. `load_eval_set` expands each YAML entry into three items (fi, en, vi) with the same expected
   pages, so the only thing that changes is the language.
2. For every strategy, retrieve once per question, then **sweep the threshold**
   (`sweep_threshold`): for t = 0.00 ... 1.00, answer rate on answerable and refusal rate on
   unanswerable questions; take the middle of the widest run of thresholds with the best
   balanced accuracy. The shipped `hybrid` setup uses `DEFAULT_MIN_RELEVANCE`, and a test fails
   if the sweep result and that constant disagree.
3. `score` per language: hit@3, hit@5, MRR, answer rate, key fact (the expected value, e.g.
   "55", is in the answer), citation validity, refusal accuracy. NaN means "nothing to measure"
   (shown as n/a), so an empty list never looks like a score of 0.
4. `cross_lingual_gap`: FI minus EN/VI MRR per strategy, and the share of the n-gram baseline's
   gap that each strategy closes.
5. `check_gate` compares the hybrid numbers with `evals/thresholds.yaml`; the CLI exits 1 on a
   miss, which fails CI.

`report.py` turns the same report into `docs/results.md` and the two charts.

## 7. Serving

`api.py: create_app` builds the assistant once (`Depends`, so tests inject a small one).
`POST /ask {"question": ..., "language": "vi"}` returns the answer, language, refusal flag,
citations and their validity, sources with scores, the Finnish query and a disclaimer.
