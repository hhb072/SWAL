CUDA_VISIBLE_DEVICES=6 python main.py --ngf=16 --ndf=64  --output_height=320  --trainroot='/home/huaibo.huang/data/rain/ddn/ddn_train' --testroot='/home/huaibo.huang/data/rain/ddn/ddn_test' --trainfiles='data/ddn_train.list' --testfiles='ddn_test.list'  --test_iter=500 --save_iter=1 --batchSize=8 --nrow=8 --lr_d=1e-4 --lr_g=1e-4  --cuda  --nEpochs=500