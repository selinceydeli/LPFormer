# Pubmed
python src/run.py --data_name pubmed --lr 1e-3  --gnn-layers 1 --dim 128 --batch-size 1024  --epochs 100 --eps 1e-5 --gnn-drop 0.3 --dropout 0.3 --pred-drop 0.3 --att-drop 0.3  --num-heads 1  --thresh-1hop 1e-2 --thresh-non1hop 1e-2 --mask-input  --feat-drop 0.3 --l2 1e-4 --eval_steps 1 --decay 1 --non-verbose --runs 1 --device 0 > pubmed_training_2.log 2>&1
