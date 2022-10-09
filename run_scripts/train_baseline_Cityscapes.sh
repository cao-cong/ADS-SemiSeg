python train_baseline.py \
--dataset cityscapes \
--gpu 0 \
--snapshot-dir ./snapshots_0.125/ \
--ignore-label 250 \
--num-classes 19 \
--input-size '256,512' \
--labeled-ratio 0.125 \
--split-id ./splits/city/split_0.pkl


