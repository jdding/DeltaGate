# Data Directory

Raw Amazon review files, processed pickle files, training shards, and
checkpoints are not in this repository. There is no download or preprocessing
script here. The paper uses three Amazon Product Reviews (2018) categories
(Video Games, CDs & Vinyl, and Movies & TV) after iterative 5-core filtering,
then the Gap-Synthesize protocol described in the paper: user-level
chronological leave-one-out, the natural gap before the held-out target,
pre-gap histories of at least five items, and the buckets `<30d`, `30-90d`,
`90-180d`, `180-365d`, and `>365d`.

Build files that match the schema below and place them in this directory
before running the training or evaluation scripts. The paper's processed
counts are recorded in `experiments/recsys2026/config_sas_public.json`.
Audited reruns also require backbone checkpoints whose SHA-256 hashes are
listed in `experiments/recsys2026/provenance/`; those checkpoints are not
released here.

Expected files:

- `amazon_cds_vinyl_processed.pkl`
- `amazon_movies_tv_processed.pkl`
- `amazon_video_games_processed.pkl`

Expected processed-data fields:

- `train_samples`: training examples, or metadata for sharded training data
- `test_samples`: full-ranking test examples
- `item_vocab_size`: item vocabulary size
- `max_len`: maximum sequence length

Expected sample fields:

- `seq_idx`: historical item ids, left-padded by the batching code
- `target_idx`: held-out target item id
- `bucket`: gap bucket label, such as `30-90d` or `>365d`
- `gap_days`: inactivity gap in days, required by DeltaGate models
- `intervals`: intra-sequence intervals, required by TiSASRec

The paper uses chronologically ordered Amazon review interactions and evaluates
returning users by inactivity-gap buckets. Raw Amazon review files are not
included in this repository.
