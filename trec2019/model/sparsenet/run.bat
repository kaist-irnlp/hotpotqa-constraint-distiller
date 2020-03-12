python sparsenet_trainer.py ^
    --n 10000 ^
    --k 1000 ^
    --data_dir ../../data ^
    -e ../../data/embedding/glove.840B.300d.gensim ^
    --gpus 0 ^
    --use_amp --amp_level O1