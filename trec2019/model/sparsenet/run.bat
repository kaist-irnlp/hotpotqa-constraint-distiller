python sparsenet_trainer.py ^
    --n 5000 10000 5000 ^
    --k 100 200 100 ^
    --data_dir ../../data ^
    -e ../../data/embedding/glove.840B.300d.gensim ^
    --gpus 0 --use_amp