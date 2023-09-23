#util.py

import pandas as pd
def update_performance_dataframe(original_df, train, test, epoch):
    train_p_r = train['precision_recall_of_role'] #precision and recall metrics of [mel, rhythm, harm], refer to model.py for details
    test_p_r = test['precision_recall_of_role']
    new_data = pd.DataFrame([[epoch, train['label_wise_acc'], test['label_wise_acc'],\
                                train['data_wise_acc'], test['data_wise_acc'],\
                                train_p_r[0], train_p_r[1], test_p_r[0], test_p_r[1],\
                                train_p_r[2], train_p_r[3], test_p_r[2], test_p_r[3],\
                                train_p_r[4], train_p_r[5], test_p_r[4], test_p_r[5]]],\
                                columns=['epoch', 'train_acc_l', 'test_acc_l', 'train_acc_d', 'test_acc_d', \
                            'train_precision_mel', 'train_recall_mel', 'test_precision_mel', 'test_recall_mel',\
                            'train_precision_rhythm', 'train_recall_rhythm', 'test_precision_rhythm', 'test_recall_rhythm',\
                            'train_precision_harm', 'train_recall_harm', 'test_precision_harm', 'test_recall_harm'] )
    return pd.concat([original_df, new_data], ignore_index=True)

   
