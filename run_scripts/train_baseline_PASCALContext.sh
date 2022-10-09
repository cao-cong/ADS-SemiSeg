python train_baseline.py \
--dataset pascal_context \
--random-mirror \
--random-scale \
--gpu 0 \
--snapshot-dir ./pc_snapshots_0.125/ \
--labeled-ratio 0.125 \
--ignore-label -1 \
--num-classes 60 \
--split-id ./splits/pc/split_0.pkl


