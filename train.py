#train.py

#########################################################################
########## Section 1. Parameters that can be modified by users ##########
#########################################################################

''' configuraion of differnt length of context and input channels '''
CONTEXT_MEASURES = 2 # i.e., "m" in paper
OTHER_INST = True #
BLEND_MODE = 'SUM' # if OTHER_INST is True #'SUM': sum all tracks => 2 channels, 'COMB': choose k-1 other trakcs => k channels
k = 4 # if BLEND_MODE is 'COMB'
TEST_DATA = [0, 4, 9]

''' CNN model hyperparameters , default is (15, 15, 10, 15) '''
CONV1_CH = 15
CONV2_CH = 15
CONV3_CH = 10
HIDDEN = 15

''' training parameters '''
lr = 0.01
BATCH_SIZE = 512
epoch_beg = 0
epoch_end = 151
experiment_name = 'exp_name'
pretrained = None
save_path = './model/'
confusion_matrix_save_path = './result/confusion_matrix/'
performance_log_save_path = './result/csv/'
meta_csv_file = './dataset/new_metadata.csv'

from data_processing.PianoRollsDataset import PianoRollsDataset
import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from model import CnnModel
from model import test_model
from util import update_performance_dataframe

#########################################################################
############## Section 2. Code for training and validation ##############
#########################################################################

''' Step 1. prepare data '''
train_set = PianoRollsDataset( meta_csv_file, test=False, test_piece=TEST_DATA, context=CONTEXT_MEASURES,\
                                other_inst=OTHER_INST, blend=BLEND_MODE, k=k )
test_set = PianoRollsDataset( meta_csv_file, test=True, test_piece=TEST_DATA, context=CONTEXT_MEASURES,\
                                other_inst=OTHER_INST, blend=BLEND_MODE, k=k )
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

''' Step 2. Build Model and optimizer '''
if OTHER_INST:
    if BLEND_MODE=='SUM': input_ch=2
    elif BLEND_MODE=='COMB': input_ch=k
else:
    input_ch=1
model = CnnModel(input_channels=input_ch, context_measures=CONTEXT_MEASURES, \
                conv1_channels=CONV1_CH, conv2_channels=CONV2_CH, conv3_channels=CONV3_CH, hidden=HIDDEN)
if pretrained is not None:
    model.load_state_dict(torch.load(pretrained))

print(model.parameters())
loss_func = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters())
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device = ', device)
model.to(device)

''' Step 3. Training iterations '''
performance = pd.DataFrame( columns=['epoch', 'train_acc_l', 'test_acc_l', 'train_acc_d', 'test_acc_d', \
                            'train_precision_mel', 'train_recall_mel', 'test_precision_mel', 'test_recall_mel',\
                            'train_precision_rhythm', 'train_recall_rhythm', 'test_precision_rhythm', 'test_recall_rhythm',\
                            'train_precision_harm', 'train_recall_harm', 'test_precision_harm', 'test_recall_harm'] )

for epoch in range(epoch_beg, epoch_end):
    print('Epoch {:4d}'.format(epoch))
    for i, (x, y_gt) in enumerate(train_loader):
        #print('x.shape = ', x.shape, ', y_gt.shape = ',y_gt.shape)
        x = x.to(device)
        y_gt = y_gt.to(device)
        
        y_pred = model(x)
        loss = loss_func(y_pred, y_gt)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
    if epoch%30==0:
        model.eval()
        # eval on both train and test set
        print('Epoch {} Test on train set'.format(epoch))
        test_on_train_info_dict = test_model(model, train_loader)
        print('Epoch {} Test on validation set'.format(epoch))
        test_on_test_info_dict = test_model(model, test_loader)
        
        # performace log
        performance = update_performance_dataframe(performance, test_on_train_info_dict, test_on_test_info_dict, epoch)
        np.save(confusion_matrix_save_path+experiment_name+'_epoch'+str(epoch)+'_train.npy', test_on_train_info_dict['con_mat'])
        np.save(confusion_matrix_save_path+experiment_name+'_epoch'+str(epoch)+'_test.npy', test_on_test_info_dict['con_mat'])
        
        # save model
        torch.save(model.state_dict(), save_path+'{}_epoch{}.pt'.format(experiment_name, epoch)) 
        model.train()
    
# end of training, save performance log
performance.to_csv(performance_log_save_path+experiment_name+'.csv')
