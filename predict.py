from myapi import *

ap = argparse.ArgumentParser(description='predict_file')

ap.add_argument('input_img', default='../aipnd-project/flowers/test/1/image_06743.jpg', nargs='*', action="store", type = str)
ap.add_argument('--top_k', default=5, dest="top_k", action="store", type=int)
ap.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
ap.add_argument('--cpu', default="cpu", action="store", dest="cpu")
ap.add_argument('--arch', dest="arch", action="store", default="vgg19", type = str)
ap.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.001)

pa = ap.parse_args()
top_k = pa.top_k
structure = pa.arch
power = pa.cpu
input_img = pa.input_img
lr = pa.learning_rate


#dataloaders , class_names, dataset_sizes = load_data()
print('load model...')

model, optimizer, criterion = set_net(structure,lr,power)

model = load_model(model, 'model.pt')
print('model succefully loaded')

print('open json file for mapping...')
with open('../aipnd-project/cat_to_name.json', 'r') as json_file:
    cat_to_name = json.load(json_file)
print('operation json ok')

print('load classes to display from image dataset')
dataloaders , class_names, dataset_sizes, image_datasets = load_data()
print('loaded')

predict_image(image_datasets, model, cat_to_name, input_img='../aipnd-project/flowers/test/1/image_06743.jpg', top_k=5, power='cpu')