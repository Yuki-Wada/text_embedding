$curr_dir = ($pwd).Path
cd I:\Yuki\workspace\mltools

pipenv run python "examples/train_cifar10_classifier.py" `
    "--train_image_npy_path" "data/preprocess/CIFAR-10/train_image.npy" `
    "--train_label_npy_path" "data/preprocess/CIFAR-10/train_label.npy" `
    "--test_image_npy_path" "data/preprocess/CIFAR-10/test_image.npy" `
    "--test_label_npy_path" "data/preprocess/CIFAR-10/test_label.npy" `
    "--output_dir_format" "data/model/cifar10/{date}" `
    "--model_name_format" "epoch-{epoch}.h5" `
    "--optimizer" "sgd" "-lr" "1e-1" "--momentum" "0.7" "--nesterov" "--clipnorm" "1e-1" `
    "--lr_decay_rate" "0.31" `
    "--lr_decay_epochs" "3" "6" "51" "52" "76" "77" "91" "92" `
    "--mb_size" "32" `
    "--epochs" "100" `
    "--seed" "2"

cd $curr_dir
