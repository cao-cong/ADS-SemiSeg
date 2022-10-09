python train_ADS_DGW.py \
--dataset cityscapes \
--snapshot-dir ./snapshots_0.125/ \
--labeled-ratio 0.125 \
--consistency_scale 10.0 \
--stabilization_scale 100.0 \
--consistency-rampup 800 \
--stabilization-rampup 800 \
--batch-size 4 \
--num-step 80000 \
--ignore-label 250 \
--num-classes 19  \
--input-size '256,512' \
--split-id ./splits/city/split_0.pkl \
--stable-threshold 0.8 \
--threshold-st 0.9 \
--learning-rate-D 1e-5 











