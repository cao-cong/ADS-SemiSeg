python train_MT_DGW.py \
--dataset pascal_voc \
--random-mirror \
--random-scale \
--snapshot-dir ./snapshots_0.125/ \
--labeled-ratio 0.125 \
--consistency 100.0 \
--consistency-rampup 800 \
--batch-size 5 \
--num-step 40000 


