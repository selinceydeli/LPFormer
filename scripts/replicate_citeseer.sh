# Citeseer
python src/run.py --data_name citeseer --lr 5e-3  --gnn-layers 1 --dim 256 --batch-size 1024  --epochs 100 --kill_cnt 100 --eps 1e-7 --gnn-drop 0.1 --dropout 0.1 --pred-drop 0.1 --att-drop 0.1  --num-heads 1  --thresh-1hop 1e-2 --thresh-non1hop 1 --feat-drop 0.1 --eval_steps 1 --decay 0.95 --non-verbose --l2 0 --runs 10 --device 0
