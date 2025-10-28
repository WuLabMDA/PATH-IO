from fastai.vision.all import *
import pandas as pd
from scipy.special import softmax

def get_results(learn, dl):

    probs, _ = learn.get_preds(dl=test_dl)
    _, pred = torch.max(probs, 1)
    probs_softmax = softmax(probs.numpy(), axis=1)

    df = pd.DataFrame(
        data={
            'fname': [file.name for file in test_dl.items],
            'probs': probs_softmax.tolist(),
            'prob_max': probs_softmax.max(axis=1).tolist(),
            'pred': pred.tolist(),
        },
        columns=['fname', 'probs', 'prob_max', 'pred']
    )

    get_id = lambda x: x.split('_')[0]   
    df['ids'] = df['fname'].map(get_id)

    get_x = lambda x: int(x.split('_')[3])
    get_y = lambda x: int(x.split('_')[6])
    df['x'] = df['fname'].map(get_x)
    df['y'] = df['fname'].map(get_y)

    get_max_x = lambda x: int(x.split('_')[4])
    get_max_y = lambda x: int(x.split('_')[7][:-4])
    df['max_x'] = df['fname'].map(get_max_x)
    df['max_y'] = df['fname'].map(get_max_y)

    return df


slide_dir = '' ## path to slide directory containg .svs files##
slide_list = sorted([ele for ele in os.listdir(slide_dir) if os.path.isdir(os.path.join(slide_dir, ele))])

for idx, cur_slide in enumerate(slide_list):
    cur_slide_name = os.path.splitext(cur_slide)[0]
    print(cur_slide_name)
    pred_dir = os.path.join('', cur_slide_name) ## path to the prediction directory where output to be saved ##
    if os.path.exists(pred_dir):
        print('Slide %s is already processed. Skipping.' % cur_slide_name)
        continue
    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir)
    cur_slide_path = os.path.join(slide_dir, cur_slide)
    dblock = DataBlock(blocks=(ImageBlock, CategoryBlock),
                   get_items=get_image_files,
                   get_y=parent_label)
    dls = dblock.dataloaders(cur_slide_path)
#     test_tfms = [ToTensor(), Resize(224), Normalize.from_stats(*imagenet_stats)]
    test_tfms = [ToTensor(), Normalize.from_stats(*imagenet_stats)]
    test_dl = dls.test_dl(get_image_files(cur_slide_path), with_labels=False, tfms=test_tfms)
    learn = load_learner('')## path to the model to perform the prediction
    learn.dls.cuda()
    df_immuno = get_results(learn, dl=test_dl)
    df_immuno.to_csv(os.path.join(pred_dir +'/' + os.path.basename(cur_slide)+'_prediction_256.csv'))






