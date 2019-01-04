python ../mmdetection/tools/test.py cfg_retinanet_ghm_r50_fpn_1x.py ghm/epoch_12.pth --gpus 8 --out results.pkl --eval bbox 2>&1 | tee ghm/eval_result.log
