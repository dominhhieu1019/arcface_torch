import argparse

import cv2
import numpy as np
import torch
import pickle
import os
from tqdm import tqdm

from backbones import get_model

def load_data(dataset):
    with open(dataset, "rb") as file:
        X, y, glasses, image_name, labels_dict = pickle.load(file)
        return X, y, glasses, image_name, labels_dict

@torch.no_grad()
def inference(weight, network, dataset, embeddings):
    if dataset is None:
        return
    
    f = open(embeddings, "wb")
    
    # if args.train:
    #     X, y, glasses, image_name, labels_dict = load_data('DB_AsianFace_face_mask/0_train.pkl')
    # else:
    X, y, glasses, image_name, labels_dict = load_data(dataset)

    # Grab the paths to the input images in our dataset
    print("[INFO] quantifying faces...")
    imagePaths = [os.path.join('DB_AsianFace_face_mask/DB_AsianFace_face_mask', '_'.join(x.split('_')[1:-1]), x) for x in image_name]
    
    # Initialize the faces embedder
    net = get_model(network, fp16=False)
    net.load_state_dict(torch.load(weight))
    # net.load_state_dict(torch.load(weight, map_location=torch.device('cpu')))  # CPU usage    
    net.eval()

    # Initialize our lists of extracted facial embeddings and corresponding people names
    knownEmbeddings = []
    knownNames = []

    for (i, imagePath) in tqdm(enumerate(imagePaths)):
        # extract the person name from the image path
        name = labels_dict[imagePath.split(os.path.sep)[-2]]
        
        # load the image
        img = cv2.imread(imagePath)
        img = cv2.resize(img,(112,112), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).unsqueeze(0).float()
        feat = net(img).numpy()

        knownNames.append(name)
        knownEmbeddings.append(feat)
    
    # print(feat)
    # print(feat.shape)

    # save to output
    data = {"embeddings": knownEmbeddings, "names": knownNames}
    f = open(embeddings, "wb")
    f.write(pickle.dumps(data))
    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')
    parser.add_argument('--network', type=str, default='r2060', help='backbone network')
    parser.add_argument('--weight', type=str, default='models/ms1mv3_arcface_r2060/backbone.pth')
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument("--embeddings", default="outputs/embeddings.pickle")
    args = parser.parse_args()
    inference(args.weight, args.network, args.dataset, args.embeddings)
