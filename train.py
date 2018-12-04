from myapi import *
parser = argparse.ArgumentParser(description='train_file')


parser.add_argument('data_dir', nargs='*', action="store", default="../aipnd-project/flowers")
parser.add_argument('--cuda', dest="cuda", action="store", default="cuda")
parser.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.001)
parser.add_argument('--epochs', dest="epochs", action="store", type=int, default=1)
parser.add_argument('--arch', dest="arch", action="store", default="vgg19", type = str)


pa = parser.parse_args()

where = pa.data_dir
lr = pa.learning_rate
structure = pa.arch
power = pa.cuda
epochs = pa.epochs


dataloaders , class_names, dataset_sizes, image_datasets = load_data(where)

model, optimizer, criterion = set_net(structure,lr,power)

train_and_save(model, criterion, optimizer, dataloaders, epochs, power)