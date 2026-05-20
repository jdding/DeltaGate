# Data Directory

Place processed Amazon review data files in this directory before running the
training or evaluation scripts.

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
