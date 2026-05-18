import os
import pandas as pd

from fastai.vision.all import *
from scipy.special import softmax


def get_results(learn, dl):

    probs, _ = learn.get_preds(dl=dl)

    _, pred = torch.max(probs, 1)

    probs_softmax = softmax(probs.numpy(), axis=1)

    df = pd.DataFrame(
        data={
            'fname': [file.name for file in dl.items],
            'probs': probs_softmax.tolist(),
            'prob_max': probs_softmax.max(axis=1).tolist(),
            'pred': pred.tolist(),
        },
        columns=['fname', 'probs', 'prob_max', 'pred']
    )

    # ---------------------------------------------------------
    # Extract slide ID
    # ---------------------------------------------------------

    get_id = lambda x: x.split('_')[0]
    df['ids'] = df['fname'].map(get_id)

    # ---------------------------------------------------------
    # Extract x and y coordinates
    # ---------------------------------------------------------

    get_x = lambda x: int(x.split('_')[3])
    get_y = lambda x: int(x.split('_')[6])

    df['x'] = df['fname'].map(get_x)
    df['y'] = df['fname'].map(get_y)

    # ---------------------------------------------------------
    # Extract maximum x and y coordinates
    # ---------------------------------------------------------

    get_max_x = lambda x: int(x.split('_')[4])
    get_max_y = lambda x: int(x.split('_')[7][:-4])

    df['max_x'] = df['fname'].map(get_max_x)
    df['max_y'] = df['fname'].map(get_max_y)

    return df


if __name__ == '__main__':

    # =========================================================
    # User-defined paths
    # =========================================================

    slide_dir = '/path/to/patch_directory/'  
    # Directory containing slide-wise patch folders

    prediction_output_dir = '/path/to/prediction_output_directory/'  
    # Directory where prediction CSV files will be saved

    model_path = '/path/to/path_io_resnet34.pkl'  
    # Path to trained FastAI model

    # =========================================================
    # Load trained model
    # =========================================================

    learn = load_learner(model_path)

    learn.dls.cuda()

    # =========================================================
    # Get slide folders
    # =========================================================

    slide_list = sorted([
        ele for ele in os.listdir(slide_dir)
        if os.path.isdir(os.path.join(slide_dir, ele))
    ])

    # =========================================================
    # Process slides
    # =========================================================

    for idx, cur_slide in enumerate(slide_list):

        cur_slide_name = os.path.splitext(cur_slide)[0]

        print(f'Processing: {cur_slide_name}')

        pred_dir = os.path.join(
            prediction_output_dir,
            cur_slide_name
        )

        if os.path.exists(pred_dir):
            print(f'Slide {cur_slide_name} is already processed. Skipping.')
            continue

        os.makedirs(pred_dir, exist_ok=True)

        cur_slide_path = os.path.join(slide_dir, cur_slide)

        # -----------------------------------------------------
        # Create dataloader
        # -----------------------------------------------------

        dblock = DataBlock(
            blocks=(ImageBlock, CategoryBlock),
            get_items=get_image_files,
            get_y=parent_label
        )

        dls = dblock.dataloaders(cur_slide_path)

        test_tfms = [
            ToTensor(),
            Normalize.from_stats(*imagenet_stats)
        ]

        test_dl = dls.test_dl(
            get_image_files(cur_slide_path),
            with_labels=False,
            tfms=test_tfms
        )

        # -----------------------------------------------------
        # Perform inference
        # -----------------------------------------------------

        df_immuno = get_results(
            learn,
            dl=test_dl
        )

        # -----------------------------------------------------
        # Save predictions
        # -----------------------------------------------------

        output_csv_path = os.path.join(
            pred_dir,
            os.path.basename(cur_slide) + '_prediction_256.csv'
        )

        df_immuno.to_csv(
            output_csv_path,
            index=False
        )

        print(f'Saved predictions to: {output_csv_path}')
