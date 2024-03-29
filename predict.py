import json, nnlib, argparse
import numpy as np

ap = argparse.ArgumentParser(description='predict-file')

# CMD arguments
ap.add_argument('input_img', default='./flowers/test/1/image_06743.jpg', nargs='*', action="store", type = str)
ap.add_argument('checkpoint', default='./trained_model.pth', nargs='*', action="store",type = str)
ap.add_argument('--top_k', default=5, dest="top_k", action="store", type=int)
ap.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
ap.add_argument('--gpu', dest="gpu", action="store", default=True, type=bool)

pa = ap.parse_args()
image_path = pa.input_img
number_of_outputs = pa.top_k
gpu = pa.gpu
path = pa.checkpoint


dataloaders, image_datasets = nnlib.load_data()
model = nnlib.load_checkpoint(path)

with open('cat_to_name.json', 'r') as json_file:
    cat_to_name = json.load(json_file)


probabilities = nnlib.predict(image_path, model, number_of_outputs, gpu)

labels = [cat_to_name[str(index + 1)] for index in np.array(probabilities[1][0].cpu())]
probability = np.array(probabilities[0][0].cpu())

print("========== START OF PREDICTION. ==========")

i=0
while i < number_of_outputs:
    print(f"{labels[i]} with a probability of {probability[i]}")
    i += 1

print("========== END OF PREDICTION. ==========")