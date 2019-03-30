import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy.misc

def save_img(out_path, img):

    scipy.misc.imsave(out_path, img, format= 'png')

def get_img(src):

    img = plt.imread(src)

    height = img.shape[0]
    width = img.shape[1]

    height = height - height % 32
    width = width - width % 32

    img = scipy.misc.imresize(img, [height, width])

    if img.shape[2] > 3:
        img = img[:,:,:3]

    return img

# saves only the deconv weights from the
# given state dictionary
def get_dc_state(state):
    my_state = {name: param for name, param in state.items() if "deconv" in name}
    return my_state


# given a model with pretrained and decoder states
# update the decoder ones with the dc_state dictionary
def update_dc_state(model_state, dc_state):
    for name, param in dc_state.items():
        model_state[name] = param
    return model_state


# given model
# and dc states
# updates only the dc states (not pretrained)
def load_state_from_dc(model, dc_path):
    dc_state = torch.load(dc_path, map_location='cpu')
    new_model_dict = update_dc_state(model.state_dict(), dc_state)

    model.load_state_dict(new_model_dict)

    return model
