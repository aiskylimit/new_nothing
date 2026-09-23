source /mnt/local/uvenvs/talas-vlm-embed/bin/activate

mkdir -p time_mem

CUDA_VISIBLE_DEVICES=0 bash ./baseline_scripts/train_distill_ckd_sigreg_cls.sh 2>&1 | tee time_mem/train_distill_ckd_sigreg_cls.log
CUDA_VISIBLE_DEVICES=1 bash ./baseline_scripts/train_distill_em_sigreg_cls.sh 2>&1 | tee time_mem/train_distill_em_sigreg_cls.log
CUDA_VISIBLE_DEVICES=2 bash ./baseline_scripts/train_distill_emo_sigreg_cls.sh 2>&1 | tee time_mem/train_distill_emo_sigreg_cls.log
CUDA_VISIBLE_DEVICES=3 bash ./baseline_scripts/train_distill_mse_sigreg_cls.sh 2>&1 | tee time_mem/train_distill_mse_sigreg_cls.log
CUDA_VISIBLE_DEVICES=4 bash ./baseline_scripts/train_distill_rkd_sigreg_cls.sh 2>&1 | tee time_mem/train_distill_rkd_sigreg_cls.log
CUDA_VISIBLE_DEVICES=5 bash ./baseline_scripts/train_distill_span_attn_cls.sh 2>&1 | tee time_mem/train_distill_span_attn_cls.log
CUDA_VISIBLE_DEVICES=6 bash ./script_full/train_distill_sigreg_cls_time_mem.sh 2>&1 | tee time_mem/train_distill_sigreg_cls_time_mem.log
