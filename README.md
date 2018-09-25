# Flower Calssifier
Built using PyTorch Framework, and available as an API.

Train a model on your own dataset by using the `train.py` python file. For inference use `predict.py` file.

All the required packages are provided in `requirements.txt` file. `model.py` and `cat_to_name.json` are supporting files.

**Examples**
$python train.py data_directory --gpu
$python predict.py image_path checkpoint

## train.py

usage: train.py [-h] [-sd SAVE_DIR] [-a {vgg16,resnet18,alexnet,squeezenet1_0,densenet161}] [-lr LEARNING_RATE] [-bs BATCH_SIZE] [-e EPOCHS] [--gpu] [--hidden_units HIDDEN_UNITS] [-nw NUM_WORKERS]
                data_directory

Train a Deep Learning Model

positional arguments:  data_directory        location of the data directory

optional arguments:
  -h, --help            show this help message and exit
  -sd SAVE_DIR, --save_dir SAVE_DIR
                        location to save model checkpoints
  -a {vgg16,resnet18,alexnet,squeezenet1_0,densenet161}, --arch {vgg16,resnet18,alexnet,squeezenet1_0,densenet161}
                        use pre-trained model
  -lr LEARNING_RATE, --learning_rate LEARNING_RATE
                        set the learning rate (hyperparameter)  -bs BATCH_SIZE, --batch_size BATCH_SIZE                        set the batch size (hyperparameter)  -e EPOCHS, --epochs EPOCHS
                        set the number of training epochs
  --gpu                 flag to use GPU (if available) for training
  --hidden_units HIDDEN_UNITS
  -nw NUM_WORKERS, --num_workers NUM_WORKERS
                        set the number of workers

## predict.py

usage: predict.py [-h] [--top_k TOP_K] [--category_names CATEGORY_NAMES] [--gpu] image_path checkpoint

Predict flower name from an image

positional arguments:
  image_path            path to the image to be predicted
  checkpoint            location of the checkpointed model

optional arguments:
  -h, --help            show this help message and exit
  --top_k TOP_K         return top k most likely classes
  --category_names CATEGORY_NAMES
                        JSON file for mapping categories to real names
  --gpu                 use GPU (if available) for inference
