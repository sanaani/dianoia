# Phase 10: Analogy datasets (scoping)

Scoped 10/05/2026. Goal: records of **cross-domain borrowings with a real outcome attached** (no LLM
grader). All sizes and licenses were checked against the official page, the hosting API or the file
itself on this date unless marked *unverified*. Samples (about 67 MB total) are in
`C:\Users\shaun\dianoia-data\work\analogy\samples\`. Free disk on `C:` is 106 GB.

## Bottom line

| Source | Gives cross-domain pairs | Outcome signal | License | Where to process |
|---|---|---|---|---|
| PatentsView (Zenodo mirror) | patent→patent citations + CPC on both ends | forward citations (computed) | CC BY 4.0 | laptop (core set is ~5 GiB zipped) |
| OpenAlex | work→work citations + field/topic on both ends | `cited_by_count`, `fwci`, `citation_normalized_percentile` | CC0 1.0 | API sample on laptop; full join on AWS |
| Reliance on Science | patent→paper (science→technology) | patent forward citations via PatentsView | CC BY-NC 4.0 | laptop (1.4 GB) |
| Analogy benchmarks | hand-built analogies | **none** (human labels only) | mixed | laptop (all under 50 MB) |
| Stack Exchange | not analogies; Q&A consensus | accepted answer, score | CC BY-SA (older dumps); newer dumps forbid LLM training | laptop per site |

**No existing analogy dataset carries a real-world outcome.** The outcome-bearing data are citation
corpora, so the training records have to be built: (problem text of the citing item, distant-domain
cited item, outcome of the citing item relative to a same-domain baseline).

---

## 1. USPTO patents (PatentsView)

### Access status (blocker found, workaround found)

- patentsview.org now redirects (301) to `https://data.uspto.gov/support/transition-guide/patentsview`.
  The legacy S3 URLs (`s3.amazonaws.com/data.patentsview.org/download/*.tsv.zip`) return **403**.
- On the USPTO Open Data Portal (product `pvgpatdis`, `https://data.uspto.gov/bulkdata/datasets/pvgpatdis`)
  sign-in with a USPTO.gov account has been required since 06/18/2026, and an API key requires
  **ID.me identity verification** (endpoint `GET https://api.uspto.gov/api/v1/datasets/products/files/{productIdentifier}/{fileName}`,
  header `X-API-KEY`, max 20 downloads of the same file per year). An anonymous call returned
  `{"message":"Unauthorized"}`. Only Shaun can do the account and ID.me steps.
- **Workaround, no account needed:** the USPTO itself deposited the PatentsView final release
  (data as of 12/31/2024) on Zenodo under CC BY 4.0:
  - Metadata tables: <https://zenodo.org/records/15783125> (72 files)
  - Claims full text: <https://zenodo.org/records/15062183> (73 files, 17.2 GiB total)
  - Detail description text: <https://zenodo.org/records/15062212>
  - File URL pattern: `https://zenodo.org/api/records/<id>/files/<name>/content`. HTTP range
    requests work (used to sample the first 2 MB of each table).
  - Creator on all three records: "United States Patent and Trademark Office".
- The ODP copy is newer (the last legacy page, archived 11/16/2025, showed data through 06/30/2025,
  released 09/10/2025). Six extra months of grants is not worth the ID.me step for phase 10.

### License

- Legacy PatentsView page (archived 11/16/2025): "All content is licensed under a Creative Commons
  Attribution 4.0 License." <https://web.archive.org/web/20251116173406/https://patentsview.org/download/data-download-tables>
- Zenodo records: license id `cc-by-4.0`. Attribution to PatentsView / USPTO is required.

### Tables needed

Format: TSV, tab-separated, double-quoted non-numeric fields, one file per zip. Zip sizes are from the
Zenodo record (12/31/2024); row counts and TSV sizes are from the legacy page (06/30/2025 release),
so the Zenodo rows are slightly fewer.

| Need | File | Zip | TSV | Rows | Columns (verified in sample) |
|---|---|---|---|---|---|
| US citations | `g_us_patent_citation.tsv.zip` | 2,056.6 MiB | 9.9 GiB | 149,301,482 | patent_id, citation_sequence, citation_patent_id, citation_date, record_name, wipo_kind, citation_category ("cited by examiner" / "cited by applicant" / ...) |
| Foreign citations | `g_foreign_citation.tsv.zip` | 657.1 MiB | 2.6 GiB | 44,145,644 | patent_id, citation_sequence, citation_application_id, citation_date, citation_category, citation_country |
| CPC per patent | `g_cpc_current.tsv.zip` | 462.7 MiB | 3.0 GiB | 57,141,640 | patent_id, cpc_sequence, cpc_section, cpc_class, cpc_subclass, cpc_group, cpc_type ("inventional"/"additional") |
| CPC titles | `g_cpc_title.tsv.zip` | 6.1 MiB | 105.5 MiB | 269,299 | subclass/group/class codes + titles (downloaded in full) |
| Grant date, title | `g_patent.tsv.zip` | 212.9 MiB | 1.0 GiB | 9,256,345 | patent_id, patent_type, patent_date, patent_title, wipo_kind, num_claims, withdrawn, filename |
| Abstracts | `g_patent_abstract.tsv.zip` | 1,571.6 MiB | 5.7 GiB | 9,256,345 | patent_id, patent_abstract |
| Claims (optional) | `g_claims_1976.tsv.zip` … `g_claims_2024.tsv.zip` | 62.7–380.6 MiB per year, 17.2 GiB total | ~2.4 GiB per recent year | — | patent_id, claim_sequence, claim_text, dependent, claim_number, exemplary |
| Coarse tech field (optional) | `g_wipo_technology.tsv.zip` | 49.7 MiB | 737 MiB | 13,453,632 | WIPO 35-field technology map |
| Science citations (optional) | `g_other_reference.tsv.zip` | 3,901.3 MiB | 9.1 GiB | 63,132,611 | free-text non-patent references |

Coverage: grants since 1976. There is **no forward-citation table**; compute it as
`count(*) group by citation_patent_id` over `g_us_patent_citation`.

### Outcome signal

- **Forward citations** of the citing patent within a fixed window (e.g., 5 years after grant),
  normalized against patents in the same CPC subclass and grant year. This is a real-world outcome,
  but a noisy one: it measures influence, not whether the borrowed mechanism worked.
- Stronger options for later: maintenance-fee payment events (the patent was kept alive) and
  litigation. Both are on the ODP behind the same account and ID.me wall; not scoped further.

### Extracting cross-domain pairs

1. Primary CPC per patent: `g_cpc_current` where `cpc_sequence = 0` (or first `inventional` row).
2. Join `g_us_patent_citation` on both `patent_id` and `citation_patent_id`.
3. Distance: different `cpc_section` (A–H, Y) = far; same section, different `cpc_class` = mid; same
   subclass = near (the baseline). A finer distance is the co-citation distance between subclasses.
4. Prefer `citation_category = "cited by applicant"`: examiner citations record prior art the examiner
   found, not something the inventor borrowed. That split is itself a useful control.
5. Record = (citing abstract/claim 1 as the problem, cited abstract as the source mechanism, CPC
   distance, normalized forward-citation outcome of the citing patent).
- Gaps: cited patents granted before 1976 have no CPC row here. `g_foreign_citation` has **no CPC
  for the cited foreign document**, so foreign citations cannot be scored for distance from
  PatentsView alone (a source with foreign CPC, such as Google Patents Public Data on BigQuery,
  was *not verified* in this pass).

---

## 2. OpenAlex

### What it gives (verified in a snapshot sample and a live API call)

Work fields include `referenced_works` (list of OpenAlex IDs), `referenced_works_count`,
`primary_topic` (topic → subfield → field → domain), `topics`, `publication_year`,
`cited_by_count`, `counts_by_year`, `fwci`, `citation_normalized_percentile`,
`cited_by_percentile_year`, `abstract_inverted_index`, `is_retracted`, `related_works`.
Hierarchy: 4 domains, 26 fields, about 250 subfields, about 4.5K topics.

### License

`s3://openalex/LICENSE.txt` is the full text of "CC0 1.0 Universal"
(<https://openalex.s3.amazonaws.com/LICENSE.txt>). No attribution needed.

### Abstracts

Stored only as `abstract_inverted_index` (word → positions); rebuild by sorting positions.
Coverage is incomplete and falling for closed-access work: OpenAlex removed Springer Nature non-OA
abstracts in 11/2022 and Elsevier closed-article abstracts from 11/2024 (reported coverage for
Elsevier 2022–2024 as low as 22.5%). In the 1,578-record sample partition, 211 had an abstract.
Use the filter `has_abstract:true` when sampling.

### Snapshot (S3)

- Bucket `s3://openalex`, region **us-east-1**, free, no AWS account needed (HTTPS:
  `https://openalex.s3.amazonaws.com/...`). Quarterly public release; current manifest is dated
  09/23/2026.
- Works only: **476,196,327 records, 2,040 files; 613.8 GiB JSONL (gz), 658.6 GiB Parquet**
  (from `data/{jsonl,parquet}/works/manifest.json`, saved in samples). The full snapshot is about
  745 GB JSONL or 770 GB Parquet, several TB uncompressed. **Does not fit the laptop.**
- Partitioned by `updated_date=YYYY-MM-DD/part_NNNN`, up to about 400K records per file (median file
  97 MB Parquet / 158 MB JSONL). Partitions are by update date, not by field, so one partition is a
  quasi-random sample, and any field-to-field join still needs an id → field map for **all** works.

### API (as of 10/2026; checked from response headers)

- No key needed for casual use, but the keyless budget is **$0.10/day**
  (`X-RateLimit-Limit-USD: 0.1`). A free key (account) gives **$1/day**. Over 100 requests/second
  triggers rate limiting. The old "polite pool" (`mailto=`) is no longer documented.
- Costs: singleton GET by ID free; list/filter $0.10 per 1,000 calls; full-text search $1 per 1,000;
  group_by $0.10 or $1 per 1,000; PDF content $10 per 1,000. `per_page` up to 100 (200 for group_by).
- Paid plans start at $5,000/year; not needed.
- Sources: <https://help.openalex.org/how-to-use-the-api/rate-limits-and-authentication>,
  <https://help.openalex.org/access/pricing/>, <https://help.openalex.org/access/example-costs/>,
  <https://help.openalex.org/download-all-data/openalex-snapshot>.

### Sampling cross-field citations without the snapshot (tested)

1. Citing sample per field and year:
   `/works?filter=primary_topic.field.id:<F>,publication_year:<Y>,referenced_works_count:>10,has_abstract:true&sample=100&seed=<s>&per_page=100&select=id,publication_year,cited_by_count,fwci,citation_normalized_percentile,referenced_works,primary_topic,abstract_inverted_index`
   (1 list call, $0.0001).
2. Resolve the references' fields in batches:
   `/works?filter=openalex:W1|W2|...&select=id,primary_topic,abstract_inverted_index&per_page=100`
   (1 call per 100 references).
3. Quick screen: `/works?filter=cited_by:<W>&group_by=primary_topic.field.id` returns the field mix
   of one work's references in a single call.
- Budget: 26 fields × 10 years × 100 works = 26K citing works × about 40 references = about
  1.05M lookups = about 10.5K list calls = **about $1.05**, so one day on a free key, or 11 days keyless.

### Outcome signal

`fwci` and `citation_normalized_percentile` are already field- and year-normalized citation impact
of the citing work. Compare works that cite a distant field against same-field works with similar
reference counts. `is_retracted` is a negative outcome.

### Full join (later, AWS)

Column-projected read of the Parquet works (id, primary_topic.field.id, publication_year,
referenced_works, fwci) with DuckDB on an EC2 instance in us-east-1. S3 → EC2 transfer in the same
region is free; a projected read should touch well under the 659 GiB total (estimate, not
measured). Output: a field-to-field citation edge table, small enough to bring home.

---

## 3. Existing analogy datasets

None has a real-world outcome. Useful as evaluation probes (does the model map structure across
domains?), not as outcome-labeled training data.

| Dataset | What | Size | License (source) | Link |
|---|---|---|---|---|
| ARN (TACL 2024) | narrative analogies: query, analogy, distractor; near/far partitions | 1.1K triples; 1.26 MB CSV | CC BY 4.0 (Zenodo) | <https://zenodo.org/records/11044026> |
| StoryAnalogy (EMNLP 2023) | story pairs with human similarity scores | 10K–100K (HF tag) | MIT (HF card) | <https://huggingface.co/datasets/JoeyCheng/story_analogy> |
| SCAR (Findings EMNLP 2023) | system-level scientific analogies with concept mappings across 13 fields | 400 analogies, 1,614 mappings | **no license file** in repo | <https://github.com/siyuyuan/scar> |
| SCAN (Findings EMNLP 2022) | scientific and metaphor analogies, full-domain mappings | under 1K (HF tag) | MIT (GitHub); "other" on HF mirror | <https://github.com/taczin/SCAN_analogies> |
| E-KAR (Findings ACL 2022) | explained analogy questions (civil-service exams) | 1K–10K per language (HF tag) | AFL-3.0 (HF card) | <https://huggingface.co/datasets/jiangjiechen/ekar_english> |
| ParallelPARC / ProPara-Logy (NAACL 2024) | process-level analogies across domains, gold and silver sets | size *unverified* | MIT (GitHub) | <https://github.com/orensul/ParallelPARC> |
| AnalogyKB | word-pair analogies from Wikidata/ConceptNet | 1,032,040 term pairs | **no license file** in repo | <https://github.com/siyuyuan/analogykb> |
| SemEval-2012 Task 2 | graded relational similarity of word pairs | 79 relation categories | CC BY 3.0 (task paper) | <https://huggingface.co/datasets/relbert/semeval2012_relational_similarity> |
| BATS, Google analogy, SAT | word analogies | 99K / 19.5K / 374 | BATS *unverified*; SAT restricted ("other" on HF) | <https://huggingface.co/datasets/relbert/analogy_questions> |
| SciMON (ACL 2024) | idea generation with retrieved "inspirations" from papers | see repo | Apache-2.0 (GitHub) | <https://github.com/EagleW/Scientific-Inspiration-Machines-Optimized-for-Novelty> |

Closest to the goal but **not public**: Hope et al. 2017 analogy mining (Quirky product ideas) and
the design-by-analogy patent corpora (Fu, Song, Murphy). They are not released as downloadable data.

**Cross-domain with an outcome, worth adding:** *Reliance on Science* (Marx and Fuegi), US patent →
scientific paper citations with applicant/examiner flag, a confidence score of 1–10, and front-page vs
in-text location. `pcs_oa_uspto.csv` is 1,371 MiB, record dated 09/25/2026, license
**CC BY-NC 4.0** (fine for private research, but no commercial use), <https://zenodo.org/records/22967630>.
The papers are OpenAlex IDs, so it bridges sources 1 and 2: science field → patent CPC, with patent
forward citations as the outcome.

---

## 4. Stack Exchange (optional)

- **Official dumps since 07/2024**: download per site after login, with an added term: the file is
  for "projects that do not include training a large language model". Dianoia trains a model, so
  **do not use the post-2024 official dumps.**
- **archive.org `stackexchange` item**: last refreshed 04/2024 (file mtimes 04/01–04/07/2024),
  99.1 GB total, 386 files, one `.7z` per site. Its `license.txt` points to CC BY-SA 3.0 with
  attribution rules (link back to question and author). Each post row carries `ContentLicense`
  (CC BY-SA 2.5 / 3.0 / 4.0). This copy predates the LLM-training clause. <https://archive.org/details/stackexchange>
- Community torrents of newer dumps exist (Academic Torrents: 12/31/2024 95.5 GB; 03/31/2025 and
  06/30/2025 about 97 GB). They inherit the new terms; avoid them.
- Relevant sites (archive.org sizes): hsm 17.3 MiB (sampled), engineering 41.5, biology 78.7,
  chemistry 120.3, physics 695.5, stats 568.9 MiB, math 3.46 GiB. Windows `tar.exe` extracts `.7z`.
- Signal (from the hsm sample: 4,576 questions, 1,938 with `AcceptedAnswerId`): accepted answer,
  answer `Score` spread and count, `PostLinks.xml` duplicates. That gives settled vs. contested
  labels, but no analogies.

---

## Recommended download plan

| Step | What | Where | Size | Estimated time |
|---|---|---|---|---|
| 1 | PatentsView core: `g_us_patent_citation`, `g_cpc_current`, `g_patent`, `g_patent_abstract`, `g_foreign_citation` (`g_cpc_title` already here) from Zenodo 15783125 | laptop, `dianoia-data\raw\patentsview\` | 4.9 GiB zip, about 22 GiB TSV (convert to Parquet one table at a time, then delete the TSV) | 10–30 min download + about 1 h DuckDB conversion |
| 2 | Build the cross-CPC citation table + 5-year forward-citation outcomes | laptop (DuckDB) | output under 5 GB | about 1 h |
| 3 | OpenAlex API stratified sample (26 fields × 10 years) with a free key | laptop | under 1 GB | 1 day ($1 free budget) |
| 4 | Reliance on Science `pcs_oa_uspto.csv` | laptop | 1.4 GB | 5 min |
| 5 | Analogy benchmarks (ARN, StoryAnalogy, SCAN, SCAR, E-KAR) as an evaluation set | laptop | under 50 MB | minutes |
| 6 (later) | Claims by year, only for the patents selected in step 2 | laptop for a few years; AWS for all 17.2 GiB zip (about 90 GiB TSV) | — | — |
| 7 (later) | Full OpenAlex field-to-field edge table | AWS EC2 in us-east-1, projected Parquet read | reads up to 659 GiB, output a few GB | a few hours; cost not estimated |

All per-dataset steps must follow the data procedure (manifest splits and an audit gate before
training). Split patents by grant year (train on earlier years, test on later ones) so the
forward-citation outcome never leaks.

### Blockers and decisions for Shaun

1. **USPTO ODP needs a USPTO.gov account + ID.me** for anything newer than 12/31/2024 or for
   maintenance-fee outcomes. The Zenodo mirror avoids this for phase 10.
2. **OpenAlex free key** ($1/day vs $0.10/day keyless) needs an openalex.org account, which only
   Shaun can create.
3. **Licenses:** Reliance on Science is non-commercial; SCAR and AnalogyKB have no license (treat
   as evaluation-only, do not redistribute); Stack Exchange post-2024 dumps forbid LLM training.
4. Forward citations and FWCI measure influence, not whether a borrowed mechanism worked. That is
   the weakest link in the "real outcome" requirement and needs an experiment card before training.

## Samples on disk

`C:\Users\shaun\dianoia-data\work\analogy\samples\`
- `patentsview\`: first 2 MB of each core table (`*.head.zip`, decoded `*.head.tsv`), full
  `g_cpc_title.tsv.zip`, `Granted_Data_Dictionary.pdf`, `g_claims_2024.head.tsv`
- `openalex\`: works manifests (JSONL and Parquet), one 1,578-record partition in both formats,
  `api_sample.json`
- `stackexchange\`: `hsm.stackexchange.com.7z` and the extracted `Posts.xml`
