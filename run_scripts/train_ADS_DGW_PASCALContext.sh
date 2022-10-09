python train_ADS_DGW.py \
--dataset pascal_context \
--random-mirror \
--random-scale \
--snapshot-dir ./ads_pc_snapshots_0.125/ \
--labeled-ratio 0.125 \
--consistency_scale 10.0 \
--stabilization_scale 100.0 \
--consistency-rampup 800 \
--stabilization-rampup 800 \
--batch-size 5 \
--num-step 80000 \
--stable-threshold 0.8 \
--ignore-label -1 \
--num-classes 60 \
--split-id ./splits/pc/split_0.pkl










