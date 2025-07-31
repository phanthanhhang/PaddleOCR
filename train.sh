# recommended paddle.__version__ == 2.0.0
# python3 -m paddle.distributed.launch --log_dir=./debug/ --gpus '0,1,2,3,4,5,6,7'  tools/train.py -c configs/rec/rec_mv3_none_bilstm_ctc.yml
python -m paddle.distributed.launch --gpus '0,1,2' tools/train.py \
-c configs/rec/PP-OCRv5/german_custom_PP_OCRv5_server_rec.yaml
# -o Global.pretrained_model=./PP-OCRv5_server_rec_pretrained.pdparams
