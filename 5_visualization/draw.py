"""
Multiple instance learning for cancer detection
step 1: wsi preprocess (create patch, extract features)
step 2: MIL aggregation model
step 3: select topk samples, then call cancer prediction model
step 4: create heatmap in wsi
"""

import os
import openslide
import cv2
from utils import segment_tissue, visualize_segmentation, \
    create_patches_in_tissue, visualize_stitch, extract_feature_with_coord, \
    read_yaml, WsiWriter, load_feature_model, visHeatmap
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from collections import OrderedDict
from sklearn.cluster import KMeans
import random
import time
import pandas as pd
import glob
from scipy.stats import rankdata
import oven
import h5py
import sys
sys.path.append("/home/huruizhen/mil")
import models
import warnings
warnings.simplefilter("ignore", category=FutureWarning)
import yaml
import opensdpc
# from models import MILModel
from PIL import ImageDraw
 

def extract_slide_feature(wsi_slide, level0_mpp, ol, encoder, fname):
    # segment the foreground
    contours_tissue, holes_tissue = segment_tissue(
        wsi_slide, level0_mpp, sthresh=8, sthresh_up=255, mthresh=7, close=4,
        use_otsu=True, filter_params={'a_t': 16.0, 'a_h': 4.0, 'max_n_holes': 8},
        ref_patch_size=512)

    vis_image = visualize_segmentation(
        wsi_slide, level0_mpp, contours_tissue, holes_tissue,
        color=(0, 255, 0), hole_color=(0, 0, 255), annot_color=(255, 0, 0),
        line_thickness=250, max_size=None, top_left=None, bot_right=None, custom_downsample=1,
        view_slide_only=False, number_contours=False, seg_display=True, anno_xml=None, annot_display=True)
    
    cv2.imwrite(f"{save_dir}/{fname}_seg.jpg", vis_image[:, :, ::-1])

    step_size = int((1 - ol) * patch_size)
    patch_coord = create_patches_in_tissue(
        wsi_slide, contours_tissue, holes_tissue, patch_level=patch_level, patch_size=patch_size,
        step_size=step_size, use_padding=True, contour_fn='four_pt_hard')

    print(f"patch coord: {patch_coord.shape}")
    
    stitch_img = visualize_stitch(
        wsi_slide, level0_mpp, patch_coord, downscale=128, patch_size=patch_size, patch_level=patch_level,
        is_draw_grid=True, bg_color=(0, 0, 0), alpha=-1)

    stitch_img.save(f"{save_dir}/{fname}_stitch.jpg")

    feature_bag = extract_feature_with_coord(wsi_slide, patch_coord, batch_size=feat_batch_size, verbose=1,
                                             print_every=20, custom_downsample=1, target_patch_size=encoder_patch_size,
                                             patch_size=patch_size, device=device, model=encoder)

    return feature_bag, patch_coord


def draw_heatmap(wsi_slide, coords, y_grades, mmap_dir="mmap_dir"):  
    print(f"drawing heatmap...")

    heatmap = visHeatmap(wsi_slide, y_grades, coords, vis_level=vis_level, cmap=colormap,
                         blur=blur, patch_size=(patch_size, patch_size), blank_canvas=False, alpha=alpha)

    heatmap = heatmap.convert("RGB") 
    return heatmap
    

def to_percentiles(scores):
    scores = rankdata(scores, 'average')/len(scores) 
    return scores


if __name__ == "__main__":
    # Directory of config file
    center_list = [center for center in glob.glob('./configs/*') if os.path.isdir(d)]

    for center_path in center_list:
        center_name = os.path.basename(center_path)
        config_yaml_list = glob.glob(f"{center_path}/*.yaml")
        print(f'Perform visualization of each task in the {center_name} center')
        print(f"{center_name} center tasks include {[os.path.basename(config_yaml)[:-len('.yaml')] for config_yaml in config_yaml_list]} to draw heatmap")

        for config_yaml in config_yaml_list:
            conf = read_yaml(config_yaml)
            
            project_name = conf.General.proj_name
            proj_type = conf.General.proj_type
            print(f'Task: {project_name}, Type: {proj_type}')

            df_path = conf.General.df_path 
            df = pd.read_csv(df_path) 
            seed_fold = conf.General.seed_fold 
            # general params
            wsi_dir = conf.General.wsi_dir  # Directory of svs file
            pretrained_dir = conf.General.pretrained_dir  # model pretrained weights
            image_feature_dir = conf.General.image_feature_dir  # Directory of image feature file
            patch_risk_heatmap = conf.General.patch_risk_heatmap  # patch risk heatmap
            patch_attention_heatmap = conf.General.patch_attention_heatmap  # patch attention heatmap

            # data params
            feat_batch_size = conf.Data.feat_batch_size
            sampling_strategy = conf.Data.sampling_strategy
            k_sample = conf.Data.k_sample
            patch_level = conf.Data.patch_level 
            patch_size = conf.Data.patch_size  
            encoder_patch_size = conf.Data.encoder_patch_size  
            
            # heatmap params
            blur = conf.Heatmap.blur
            colormap = conf.Heatmap.colormap
            overlay = conf.Heatmap.overlay 
            blank_canvas = conf.Heatmap.blank_canvas
            alpha = conf.Heatmap.alpha
            device = conf.General.device


            df['filename'].astype(str)
            for idx in range(len(df["filename"])):
                row = df.iloc[idx]
                """
                filename:     20445123-2.pt
                slide_id:     20445123-2
                slide_id_svs: 20445123-2.svs
                """
                df['filename'].astype(str)
                slide_filename = row.filename  #  in "../1_process_wsi/WSIs_source/"  folder
                slide_id = row.filename[:-len(".pt")] 


                slide_id_svs = row.filename.replace('.pt','.svs')  
                print(f"Slide Id: {slide_id}, Slide Id SVS: {slide_id_svs}")

                start_time = time.time()

                # wsi_name = os.path.join(wsi_dir, f"{slide_id}.ndpi")
                wsi_name = os.path.join(wsi_dir, slide_id_svs)
                print(wsi_name)
                slide = openslide.open_slide(wsi_name)

                # get the mpp
                mpp = row.resolution 
                vis_level = int(np.log2(conf.Heatmap.vis_mpp / mpp))  
                
                print(f"patch_level: {patch_level} patch_size: {patch_size} encoder_patch_size: {encoder_patch_size}")
                import models

                net = models.get_obj_from_str(conf.Model.name)  
                MIL = net(**conf.Model.params).to(device)   
                n_classes = conf.Model.params.n_classes   
                print(f"Task: {project_name}, Type: {proj_type}")
                print(f"num_classes: {n_classes}")
                if proj_type == 'cls':
                    if n_classes == 2:   # binary classification task
                        print(f"binary classification task")
                        save_dir = f"./figs/{center_name}/{project_name}/positive" if row[project_name] else f"./figs/{center_name}/{project_name}/negative"


                    elif n_classes == 3:  # Three-class classification tasks
                        print(f"three classification task, save_dir: {save_dir}")
                        save_dir = f"./figs/{center_name}/{project_name}/positive" if row[project_name] == 2 else \
                                f"./figs/{center_name}/{project_name}/middle" if row[project_name] == 1 else \
                                f"./figs/{center_name}/{project_name}/negative" if row[project_name] == 0 else None


                elif proj_type == 'reg':

                    if row['status'] == 1:  
                        print(f"slide id: {slide_id}, status: 1")
                        save_dir = f"./figs/{center_name}/{project_name}/status_1"  

                    else:
                        print(f"slide id: {slide_id}, status: 0")
                        save_dir = f"./figs/{center_name}/{project_name}/status_0"  

                os.makedirs(save_dir, exist_ok=True)


                seed,fold = seed_fold
                pretrained_MIL_weight = os.path.join(pretrained_dir, f"{project_name}.pth")
                
                new_state_dict = OrderedDict()
                print(f"Model weights path: {pretrained_MIL_weight}")
                td = torch.load(pretrained_MIL_weight, map_location="cpu")

                for key, value in td.items():
                    k = key[len("model."):]   
                    new_state_dict[k] = value


                MIL.load_state_dict(new_state_dict, strict=True)
                print(f"Model loading weight successfully")

                # generate heatmap
                feat_path = os.path.join(image_feature_dir, f"{slide_id}.pt")
                feat_dir = f"../2_extract_features/outputs/images/{slide_id}.pt"  

                coord_dir = conf.General.coord_dir 
                coord_path = os.path.join(coord_dir,f"{slide_id}.h5") 
                if os.path.exists(feat_dir) and os.path.exists(coord_dir):  
                    feature_bag = torch.load(feat_dir)
                    print(f"slide: {slide_id}, image feature shape: {feature_bag.shape}")
                    with h5py.File(coord_path, 'r') as h5_file:
                        patch_coord_overlay = np.array(h5_file['coords'])
                        if patch_coord_overlay.shape[0] > 6000:
                            patch_coord_overlay = patch_coord_overlay[:6000]
                        print(f"slide: {slide_id}, coords shape: {patch_coord_overlay.shape}")

                else:
                    raise FileNotFoundError(f"Feature file {feat_dir} or coord file {coord_path} not found. Please extract features first.")

                print(f"heatmap feature bag shape : {feature_bag.shape}")

                MIL.eval()
                with torch.inference_mode(): 
                    if proj_type == 'cls':
                        y_label = int(row[project_name]) 
                        # process by steps, patch-level probs
                        h = feature_bag.to(device)
                        A, h = MIL.model.attention_net(h.float())  
                        print(f"A.shape: {A.shape},h.shape: {h.shape}")  
                        A_raw = torch.transpose(A, 1, 0) 

                        logits = MIL.model.classifiers(h)
                        Y_probs = F.softmax(logits, dim=1)[:, y_label]   


                        # slide prob，
                        A_probs = F.softmax(A_raw, dim=1)  
                        M = torch.mm(A_probs, h) 
                        case_id = row.case_id
                        logit = MIL.model.classifiers(M)   

                        Y_prob = F.softmax(input=logit,dim=1)[:,y_label].cpu().item() # slide-level probability
                        print(f"{slide_id} predict probability：{F.softmax(logit, dim=1)}, label：{y_label}")

                        
                    elif proj_type == 'coxreg':
                        y_label = int(row[project_name]) 
                        # process by steps, patch-level probs

                        h = feature_bag.to(device)
                        print(f"feature_bag: {h.shape}")
                        A, h = MIL.model.attention_net(h.float()) 
                        print(f"A.shape: {A.shape},h.shape: {h.shape}")  
                        A_raw = torch.transpose(A, 1, 0)  
                        
                        logits = MIL.model.classifiers(h)  
                        print(f"logits: {logits[:]}")
                        Y_probs = (logits - logits.min()) / (logits.max() - logits.min())
                        print(f"Y_probs: {Y_probs}\nY_probs.shape: {Y_probs.shape}")


                        A_probs = F.softmax(A_raw, dim=1)  
                        M = torch.mm(A_probs, h)  
                        print(f"M.shape: {M.shape}")
                        case_id = row.case_id
                        logit = MIL.model.classifiers(M)  
                        print(f"slide logit.shape: {logit.shape},logit: {logit}")
                        risk_score  = logit.cpu().item()
                        Y_prob = risk_score
                        print(f"slide-level Risk score : {Y_prob}")



                if proj_type == 'cls':
                    A_raw = A_raw.reshape(-1).cpu()
                    Y_probs = Y_probs.reshape(-1).cpu().numpy()
                    A_norm = to_percentiles(A_raw)
                    y_risks = A_norm * Y_probs 

                    # !  Visualization using patch risk
                    if patch_risk_heatmap:
                        # print(f"对{slide_id}进行patch级别的可视化")
                        heatmap = draw_heatmap(slide, patch_coord_overlay, y_risks)
                        os.makedirs(f"{save_dir}/patch_risk_heatmap/", exist_ok=True)

                        heatmap.save(f"{save_dir}/patch_risk_heatmap/{slide_id}_{Y_prob:.3f}.png")
                        print(f"total time elapsed: {(time.time() - start_time):.2f}")

                    # ! Visualization using patch attention
                    if patch_attention_heatmap:
                        heatmap = draw_heatmap(slide, patch_coord_overlay, A_norm)
                        os.makedirs(f"{save_dir}/patch_attention_heatmap/", exist_ok=True)

                        heatmap.save(f"{save_dir}/patch_attention_heatmap/{slide_id}_{Y_prob:.3f}.png")
                        print(f"total time elapsed: {(time.time() - start_time):.2f}")


                elif proj_type == 'coxreg':
                    A_raw = A_raw.reshape(-1).cpu()
                    Y_probs = Y_probs.reshape(-1).cpu().numpy()
                    A_norm = to_percentiles(A_raw)
                    y_risks = A_norm 

                    heatmap = draw_heatmap(slide, patch_coord_overlay, y_risks)
                    heatmap.save(f"{save_dir}/{row['status']}_{slide_id}_{Y_prob:.3f}.png")

                    print(f"total time elapsed: {(time.time() - start_time):.2f}")
