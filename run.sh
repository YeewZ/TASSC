python train.py --gpu 0 --times 5 --epochs 200 --dropout_ratio 0.1 --lr 0.01 --domain Citation --source ACMv9      --target Citationv1 --cons_par 1.0 --tau  0.5 --weight_decay 0.005 --tgt_prop 10 --bottleneck 128
python train.py --gpu 0 --times 5 --epochs 200 --dropout_ratio 0.1 --lr 0.01 --domain Citation --source ACMv9      --target DBLPv7     --cons_par 1.0 --tau 0.25 --weight_decay 0.005 --tgt_prop 10 --bottleneck 128
python train.py --gpu 0 --times 5 --epochs 200 --dropout_ratio 0.1 --lr 0.01 --domain Citation --source Citationv1 --target ACMv9      --cons_par 1.0 --tau 0.25 --weight_decay 0.005 --tgt_prop 10 --bottleneck 128
python train.py --gpu 0 --times 5 --epochs 200 --dropout_ratio 0.1 --lr 0.01 --domain Citation --source Citationv1 --target DBLPv7     --cons_par 1.0 --tau 0.25 --weight_decay 0.005 --tgt_prop 10 --bottleneck 128
python train.py --gpu 0 --times 5 --epochs 200 --dropout_ratio 0.1 --lr 0.01 --domain Citation --source DBLPv7     --target ACMv9      --cons_par 1.0 --tau 0.25 --weight_decay 0.005 --tgt_prop 10 --bottleneck 128
python train.py --gpu 0 --times 5 --epochs 200 --dropout_ratio 0.1 --lr 0.01 --domain Citation --source DBLPv7     --target Citationv1 --cons_par 1.0 --tau 0.25 --weight_decay 0.005 --tgt_prop 10 --bottleneck 128

python train.py --gpu 0 --times 5 --epochs 200 --dropout_ratio 0.1 --lr 0.01 --domain Airport --source EUROPE --target USA --cons_par 1.0 --tau 1.0 --weight_decay 0.001 --tgt_prop 1  --bottleneck 128
python train.py --gpu 0 --times 5 --epochs 200 --dropout_ratio 0.1 --lr 0.01 --domain Airport --source USA --target EUROPE --cons_par 1.0 --tau 1.0 --weight_decay 0.001 --tgt_prop 10 --bottleneck 128

python train.py --gpu 0 --times 5 --epochs 200 --dropout_ratio 0.1 --lr 0.01 --domain Twitch --source EN --target DE --cons_par 1.0 --tau 1.0 --weight_decay 0.005 --tgt_prop 1  --bottleneck 64
python train.py --gpu 0 --times 5 --epochs 200 --dropout_ratio 0.1 --lr 0.01 --domain Twitch --source DE --target EN --cons_par 0.5 --tau 0.5 --weight_decay 0.005 --tgt_prop 1  --bottleneck 64
