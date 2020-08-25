import os 
import os.path as osp
import pandas as pd
import torch
import joblib
from fastai.tabular import *
from fastai.vision import *
from fastai.metrics import *
from fastai.callbacks import *
from fastai.metrics import error_rate, accuracy
import random

class ImageTabularModel(nn.Module):
    "Basic model for tabular data."
    def __init__(self, emb_szs:ListSizes, n_cont:int, layers:Collection[int], ps:Collection[float]=None):
        super().__init__()
        self.cnn = create_body(models.resnet34,pretrained = True) #resnet34 for images
        self.tab = TabularModel(emb_szs, n_cont, 128, layers, use_bn = False, emb_drop = ps) #Tabular model for metadata

        self.reduce = nn.Sequential(*([AdaptiveConcatPool2d(), Flatten()] + bn_drop_lin((1024), 512, bn=True, p=0.3, actn=nn.ReLU(inplace=True)))) #Use this FC layers to reduce nodes
        self.merge = nn.Sequential(*bn_drop_lin(512 + 128, 128, bn=True, actn=nn.ReLU(inplace=True))) #Merge 2 models together
        self.final = nn.Sequential(*bn_drop_lin(128, 20, bn=True, p=0.)) # Last FC layer for regression

    def forward(self, img:Tensor, x:Tensor) -> Tensor:
        imgLatent = self.reduce(self.cnn(img))
        tabLatent = self.tab(x[0],x[1])
        cat = torch.cat([imgLatent, tabLatent], dim=1) #re-define forward func for concat model
        merge = self.merge(cat)
        final = self.final(merge)
        return final 

class SaveBestModel(Recorder):
    def __init__(self, learn,name='best_model'):
        super().__init__(learn)
        self.name = name
        self.best_loss = None
        self.best_acc = None
        self.save_method = self.save_when_acc
        
    def save_when_acc(self, metrics):        
        loss, acc = metrics[0], metrics[1]
        if self.best_acc == None or acc > self.best_acc:
            self.best_acc = acc
            self.best_loss = loss
            self.learn.save(f'{self.name}')
            print("Save the best accuracy {:.5f}".format(self.best_acc))
        elif acc == self.best_acc and  loss < self.best_loss:
            self.best_loss = loss
            self.learn.save(f'{self.name}')
            print("Accuracy is eq,Save the lower loss {:.5f}".format(self.best_loss))
            
    def on_epoch_end(self,last_metrics=MetricsList,**kwargs:Any):
        self.save_method(last_metrics)
        
        
DATA_DIR = '/mnt/sda/hong01-data/MART_DATA/OUTPUT_MERGED'
CSV_DIR = osp.join(DATA_DIR, 'PANDAS')
IMG_DIR = osp.join(DATA_DIR, 'AUTOGRAPHER')
SCREENSHOT_DIR = osp.join(DATA_DIR, 'SCREENSHOTS')

events_text = {'act00': 'calibration',
                'act01': 'write an email',
                'act02': 'read on screen',
                'act03': 'edit/create presentation',
                'act04': 'zone out and fixate',
                'act05': 'use a calculator to add up numbers on sheet',
                'act06': 'physical precision task',
                'act07': 'put documents in order',
                'act08': 'read text/numbers on page',
                'act09': 'arrange money in change jar',
                'act10': 'write on paper with pen',
                'act11': 'watch a youtube video',
                'act12': 'go to a news website and browse',
                'act13': 'have conversation with experimenter in room',
                'act14': 'make a telephone call',
                'act15': 'drink/eat for 2 minutes',
                'act16': 'close eyes and sit still',
                'act17': 'clean e.g. sweaping the floor, wipe, ...',
                'act18': 'exercise: sit up/stand down repeatedly',
                'act19': 'hand-eye coordination (tennis ball)',
                'act20': 'pace the room',
                }


##### DATA PREPARATION #####
print('===== DATA PREPARATION =====')
dfA = pd.read_csv(osp.join(CSV_DIR, 'trainA.csv'), index_col=0)
dfB = pd.read_csv(osp.join(CSV_DIR, 'trainB.csv'), index_col=0)
df = pd.concat([dfA, dfB])
df = df.sort_values(by=['event_id', 'sub_id'])

imgs = sorted(os.listdir(IMG_DIR))
imgs = [item[:5]+'test_'+item[5:] if 'pred' in item else item for item in imgs]

df_image = pd.DataFrame([item.split('_')+[item] for item in imgs], columns=['sub_id', 'source', 'event_id', 'img_order', 'image_path'])
df_image = df_image.astype({'sub_id': 'int64'})

df_train = pd.merge(df, df_image, how='inner', on=['sub_id', 'event_id', 'source'])
df_train['label'] = df_train.apply(lambda row: int(row['event_id'][-2:]), axis=1)


##### TRAINING AND PREDICTING #####
print('===== TRAINING AND PREDICTING =====')
valid_pct = 0.2

valid_index = []
start_idx = 0

for idx, key in enumerate(events_text.keys()): 
    num_entries = df_train[df_train['event_id']==key].shape[0]
    num_valid_idx = round(num_entries * valid_pct)
    idx = random.sample(range(start_idx, start_idx+num_entries), num_valid_idx)
    valid_index += idx
    start_idx += num_entries
    
dep_var = 'label'
cat_names = []

data_columns = df_train.columns.str.startswith("data_")
data_autographer = [not item for item in df_train.columns.str.startswith("data_AUTOGRAPHER")]
data_mouse = [not item for item in df_train.columns.str.startswith("data_MOUSE")]
cont_names = list(df_train.loc[:, data_columns & data_autographer].columns)

procs = [FillMissing, Categorify, Normalize]

train_tabular_list = TabularList.from_df(df_train, cont_names=cont_names, procs=procs, path='./')
train_image_list = ImageList.from_df(df_train, path=IMG_DIR, cols='image_path')

train_mixed_list = (MixedItemList([train_image_list, train_tabular_list], path='./', inner_df=train_tabular_list.inner_df)
.split_by_idx(valid_index)
.label_from_df(cols=dep_var, label_cls=CategoryList))

data = train_mixed_list.databunch(bs=4)
emb = data.train_ds.x.item_lists[1].get_emb_szs()
model = ImageTabularModel(emb, 0, [1000,500], ps=0.2)

learn = Learner(data, model, metrics=[accuracy], loss_func=torch.nn.CrossEntropyLoss())
#learn = Learner(data, model, metrics=[accuracy], loss_func=torch.nn.CrossEntropyLoss(), callback_fns=SaveBestModel)

# load model
print('Load model ...')
learn.load('bestmodel')

print('Start training ...')
learn.fit_one_cycle(150, 1e-3, callbacks = SaveModelCallback(learn))
#learn.fit_one_cycle(250, 1e-3)

print('Saving ...')
learn.save('stage-1')

joblib.dump(learn, 'learn.joblib')