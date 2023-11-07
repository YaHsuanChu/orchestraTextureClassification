#model.py
'''define a simple CNN model, model hyperparameters could be modified in train.py'''

import torch
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
from sklearn.metrics import confusion_matrix
import numpy as np

class CnnModel(torch.nn.Module):
    def __init__(self, input_channels=2, context_measures=0, conv1_channels=15, conv2_channels=15, conv3_channels=10, hidden=15):
        super().__init__()
        '''
            total_measure = 1+context_measures*2
            input shape = (96*total_measure, 128)
            output = (is_mel, is_rhythm, is_harm) 
        '''
        total_measures = 1+context_measures*2
        
        '''shape (in_channels, 96*total_measures, 128) -> (conv1_channels, 24*total_measures, 32)'''
        self.conv1 = torch.nn.Sequential(
                Conv2d(in_channels=input_channels, out_channels=conv1_channels, kernel_size=(3, 3), stride=(2, 2), padding=1),
                ReLU(),
                MaxPool2d(kernel_size=(2, 2), stride=(2, 2)) ) 
        
        '''shape (conv1_channels, 24*total_measures, 32) -> (conv2_channels, 12*total_measures, 8)'''
        self.conv2 = torch.nn.Sequential(
                Conv2d(in_channels=conv1_channels, out_channels=conv2_channels, kernel_size=(3, 3), stride=(1, 2), padding=1),
                ReLU(),
                MaxPool2d(kernel_size=(2, 2), stride=(2, 2)) )
       
        '''shape (conv2_channels, 12*total_measures, 8) -> (conv3_channels, 6*total_measures, 2)'''
        self.conv3 = torch.nn.Sequential( 
                Conv2d(in_channels=conv2_channels, out_channels=conv3_channels, kernel_size=(3, 3), stride=(1, 2), padding=1),
                ReLU(),
                MaxPool2d(kernel_size=(2, 2), stride=(2, 2)) )
        
        '''flatten -> shape = (12*conv3_channels*total_measures,)'''
        '''shape (12*conv3_channels*total_measures, ) -> (3, )'''
        self.out = torch.nn.Sequential(
            Linear(12*conv3_channels*total_measures, hidden),
            ReLU(),
            Linear(hidden, 3),
            torch.nn.Sigmoid() )

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = torch.flatten(x, start_dim=1)
        out = self.out(x)

        return out

def three_bool_to_eight_class(three_bool):
    '''
    input: shpae=(n, 3)
    convert to three bool value [is_mel, is_rhythm, is_harm] to 2^3=8 classes, classed number is from 0 to 7
    example.
        [0, 0, 0] => class 0: no roles
        [1, 0, 0] => class 1: melody only
        [0, 1, 1] => class 6: rhythm and harmony
    '''
    if three_bool.shape[-1]!=3: 
        raise Exception('Invalid bool array to convert to classes')
    cls = three_bool[:,0]*1 + three_bool[:,1]*2 + three_bool[:,2]*4
    return cls

def test_model(model, loader):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with torch.no_grad():
        # for overall accuracy
        total = 0.0
        correct_label_wise = 0.0
        correct_for_all_three_labels = 0.0
        
        # for precision and recall of each label
        TP = np.zeros(3, dtype=np.float32)
        pred_label_count = np.zeros(3, dtype=np.float32)
        gt_label_count = np.zeros(3, dtype=np.float32)
        
        # for predicted labels analysis including confusion matrix
        predicted_class = []
        gt_class = []

        for i, (x, y_gt) in enumerate(loader):
            x = x.to(device)
            y_gt = y_gt.to(device)
            y_pred = model(x)
            y_pred = y_pred > 0.5
            total += x.shape[0]
            same = (y_gt==y_pred)
            
            # calculate overall accuracy
            correct_label_wise += same.sum()
            number_of_correct_labels_per_data = same.sum(axis=1)
            correct_for_all_three_labels += (number_of_correct_labels_per_data==3).sum()
             
            # calculate Precision and Recall for each label
            gt_label_count += y_gt.sum(axis=0).cpu().numpy()
            pred_label_count += y_pred.sum(axis=0).cpu().numpy()
            TP += (torch.logical_and(same,y_gt)).sum(axis=0).cpu().numpy()
                
            # for confusion matrix
            predicted_class.append( three_bool_to_eight_class(y_pred) )
            gt_class.append( three_bool_to_eight_class(y_gt) )
            
        # overall accuracy
        label_wise_acc = (correct_label_wise/(total*3)).cpu().item()
        data_wise_acc = (correct_for_all_three_labels/total).cpu().item()
        print('label-wise accuracy = {:.3f}'.format(label_wise_acc))
        print('data-wise accuracy (all three labels are correct) = {:.3f}'.format(data_wise_acc))
        
        # precision and recall for each roles
        precision_mel = -1 if pred_label_count[0]==0 else TP[0]/pred_label_count[0]
        recall_mel = -1 if gt_label_count[0]==0 else TP[0]/gt_label_count[0]
        precision_rhythm = -1 if pred_label_count[1]==0 else TP[1]/pred_label_count[1]
        recall_rhythm = -1 if gt_label_count[1]==0 else TP[1]/gt_label_count[1]
        precision_harm = -1 if pred_label_count[2]==0 else TP[2]/pred_label_count[2]
        recall_harm = -1 if gt_label_count[2]==0 else TP[2]/gt_label_count[2]
        print('(Precision, Recall) of [mel, rhythm, harm] : [({:.3f}, {:.3f}), ({:.3f}, {:.3f}), ({:.3f}, {:.3f})]'.format(\
                precision_mel, recall_mel, precision_rhythm, recall_rhythm, precision_harm, recall_harm))
        
        # confusion matrix
        predicted_class = torch.concatenate(predicted_class, axis=0).cpu().numpy()
        gt_class = torch.concatenate(gt_class, axis=0).cpu().numpy()
        con_mat = confusion_matrix(gt_class, predicted_class, normalize='true')
        
        # return testing information
        info_dict = {   'label_wise_acc': label_wise_acc,\
                        'data_wise_acc': data_wise_acc,\
                        'precision_recall_of_role': (precision_mel, recall_mel,\
                                precision_rhythm, recall_rhythm, precision_harm, recall_harm),\
                        'con_mat': con_mat }
        return info_dict
