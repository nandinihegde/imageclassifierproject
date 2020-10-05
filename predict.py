
#import libraries
import argparse

import utilityfuncs
import modelling
import json
#Command Line Arguments

ap = argparse.ArgumentParser( description='Predict from Image')
ap.add_argument('input_img', default='./flowers/test/101/image_07949.jpg', nargs='*', action="store", type = str)
ap.add_argument('checkpoint', default='./checkpointsave/checkpoint.pth', nargs='*', action="store",type = str)
ap.add_argument('--top_k', default=5, dest="top_k", action="store", type=int)
ap.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
ap.add_argument('--gpu', default="gpu", action="store", dest="gpu")

print("These are the params you have set:")
print(ap.parse_args())
pa = ap.parse_args()
image_path = pa.input_img
number_of_outputs = pa.top_k
power = pa.gpu
input_img = pa.input_img
checkpoint_path = pa.checkpoint
categoryjson=pa.category_names


#image_datasets,dataloaders = utilityfuncs.load_data(datapath)

model=utilityfuncs.load_checkpoint(checkpoint_path)


with open(categoryjson, 'r') as json_file:
    cat_to_name = json.load(json_file)


prob, labels = modelling.predict(image_path, model, number_of_outputs, power)


i=0
while i < number_of_outputs:
    print("The image is of flower {}, with a probability of {}".format(labels[i], prob[i]))
    i += 1

print("Prediction is complete")