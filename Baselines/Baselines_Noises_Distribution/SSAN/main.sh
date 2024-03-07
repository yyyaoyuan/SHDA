# ===============================================================================================
# Distribution
nohup python main.py --source NDU --target TS5 --cuda 0 --nepoch 1000 --partition 10 --prototype three --layer double --d_common 256 --optimizer mSGD --lr 0.1 --alpha 0.1 --beta 0.004 --gamma 0.1 --combine_pred Cosine --checkpoint_path checkpoint/ --temperature 5.0 > NDUTS5.log 2>&1 &

nohup python main.py --source NDL --target TS5 --cuda 0 --nepoch 1000 --partition 10 --prototype three --layer double --d_common 256 --optimizer mSGD --lr 0.1 --alpha 0.1 --beta 0.004 --gamma 0.1 --combine_pred Cosine --checkpoint_path checkpoint/ --temperature 5.0 > NDLTS5.log 2>&1 &
