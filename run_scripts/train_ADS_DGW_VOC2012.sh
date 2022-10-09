python train_ADS_DGW.py \
--random-mirror \
--random-scale \
--snapshot-dir ./snapshots_0.125/ \
--labeled-ratio 0.125 \
--consistency_scale 10.0 \
--stabilization_scale 100.0 \
--consistency-rampup 800 \
--stabilization-rampup 800 \
--batch-size 5 \
--num-step 90000 \
--stable-threshold 0.8 











