cd ..

pipenv run python "examples/train_cifar10_classifier.py" \
    "--train_image_npy_path" "data/preprocess/CIFAR-10/train_image.npy" \
    "--train_label_npy_path" "data/preprocess/CIFAR-10/train_label.npy" \
    "--test_image_npy_path" "data/preprocess/CIFAR-10/test_image.npy" \
    "--test_label_npy_path" "data/preprocess/CIFAR-10/test_label.npy" \
    "--model" "res_net" \
    "--output_dir_format" "data/model/cifar10/{date}" \
    "--model_name_format" "epoch-{epoch}.h5" \
    "--optimizer" "sgd" \
    "-lr" "1e-1" \
    "--momentum" "0.7" "--nesterov" \
    "--lr_scheduler" "cosine_annealing" \
    "--lr_decay" "0.1" \
    "--lr_steps" "0.1" "0.5" "0.75" "0.9" \
    "--min_lr" "1e-5" \
    "--mb_size" "256" \
    "--epochs" "100" \
    "--seed" "2"

cd scripts
