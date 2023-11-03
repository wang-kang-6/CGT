import os
from glob import glob
import argparse
from PIL import Image
import numpy as np
from tqdm import tqdm
import torch 
from dgl.data.utils import save_graphs
import h5py

from histocartography.preprocessing import (
    VahadaneStainNormalizer,         
    NucleiExtractor,                 
    DeepFeatureExtractor,            
    KNNGraphBuilder,                 
    ColorMergedSuperpixelExtractor,  
    DeepFeatureExtractor,            
    RAGGraphBuilder,                 
    AssignmnentMatrixBuilder         
)


TUMOR_TYPE_TO_LABEL = {
    'N': 0,
    'PB': 1,
    'UDH': 2,
    'ADH': 3,
    'FEA': 4,
    'DCIS': 5,
    'IC': 6
}

MIN_NR_PIXELS = 50000
MAX_NR_PIXELS = 50000000  
STAIN_NORM_TARGET_IMAGE = '../data/target.png'  


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image_path',
        type=str,
        help='path to the BRACS data.',
        default='',
        required=False
    )
    parser.add_argument(
        '--save_path',
        type=str,
        help='path to save the TG graphs.',
        default='',
        required=False
    )
    return parser.parse_args()

class TgBuilding:

    def __init__(self):
        
        self.normalizer = VahadaneStainNormalizer(target_path=STAIN_NORM_TARGET_IMAGE)

        self._build_tg_builders()

        self.image_ids_failing = []
        
        self.knn_graph_builder = KNNGraphBuilder(k=5, thresh=50, add_loc_feats=True)

    def _build_tg_builders(self):
        
        self.tissue_detector = ColorMergedSuperpixelExtractor(
            superpixel_size=500,
            compactness=20,
            blur_kernel_size=1,
            threshold=0.05,
            downsampling_factor=4
        )
        
        self.tissue_feature_extractor = DeepFeatureExtractor(
            architecture='resnet34',
            patch_size=144,
            resize_size=224
        )

        self.rag_graph_builder = RAGGraphBuilder(add_loc_feats=True)

    def _build_tg(self, image):
        superpixels, _ = self.tissue_detector.process(image)
        features = self.tissue_feature_extractor.process(image, superpixels)
        graph = self.rag_graph_builder.process(superpixels, features)
        return graph, superpixels

    def process(self, image_path, save_path, split):
        
        subdirs = os.listdir(image_path)
        image_fnames = []
        for subdir in (subdirs + ['']):  
            image_fnames += glob(os.path.join(image_path, subdir, '*.png'))

        print('*** Start analysing {} images ***'.format(len(image_fnames)))
        for image_path in tqdm(image_fnames):

            
            _, image_name = os.path.split(image_path)
            image = np.array(Image.open(image_path))
            nr_pixels = image.shape[0] * image.shape[1]
            image_label = TUMOR_TYPE_TO_LABEL[image_name.split('_')[2]]
            tg_out = os.path.join(save_path, 'tissue_graphs', split, image_name.replace('.png', '.bin'))

            if not  self._valid_image(nr_pixels):
                
                try: 
                    image = self.normalizer.process(image)
                except:
                    print('Warning: {} failed during stain normalization.'.format(image_path))
                    self.image_ids_failing.append(image_path)
                    pass


                try: 
                    tissue_graph, tissue_map = self._build_tg(image)
                    save_graphs(
                        filename=tg_out,
                        g_list=[tissue_graph],
                        labels={"label": torch.tensor([image_label])}
                    )
                except:
                    print('Warning: {} failed during tissue graph generation.'.format(image_path))
                    self.image_ids_failing.append(image_path)
                    pass
            else:
                print('Image:', image_path, ' was already processed or is too large/small.')

        print('Out of {} images, {} successful graph generations.'.format(
            len(image_fnames),
            len(image_fnames) - len(self.image_ids_failing)
        ))
        print('Failing IDs are:', self.image_ids_failing)

    def _valid_image(self, nr_pixels):
        if nr_pixels > MIN_NR_PIXELS and nr_pixels < MAX_NR_PIXELS:
            return True
        return False

    def _exists(self,tg_out):
        if os.path.isfile(tg_out):
            return True
        return False


if __name__ == "__main__":

    args = parse_arguments()
    if not os.path.isdir(args.image_path) or not os.listdir(args.image_path):
        raise ValueError("Data directory is either empty or does not exist.")

    split = ''
    if 'train' in args.image_path:
        split = 'train'
    elif 'val' in args.image_path:
        split = 'val'
    else:
        split = 'test'

    os.makedirs(os.path.join(args.save_path, 'tissue_graphs', split), exist_ok=True)

    tg_builder = TgBuilding()
    tg_builder.process(args.image_path, args.save_path, split)
