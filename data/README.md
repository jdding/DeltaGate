# Data Directory

Place raw data files here after downloading from Amazon product data.

Expected files:
- `amazon_cds_vinyl_processed.pkl`
- `amazon_movies_tv_processed.pkl`
- `amazon_video_games_processed.pkl`

Processing steps:
1. Download Amazon product data (reviews and metadata)
2. Build user interaction sequences with timestamps
3. Split into train/test by time (last 7 days for test)
4. Filter to cold-start users with 30+d gap between train and test
5. Save as pickle with train_samples, test_samples, item_vocab_size, max_len
