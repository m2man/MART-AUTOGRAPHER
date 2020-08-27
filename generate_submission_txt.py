import joblib
import os
import numpy as np
import pandas as pd


# READ TEST.CSV
RUN = 'test' # trainA, trainB, test
PANDAS_DIR = '/mnt/sda/hong01-data/MART_DATA/OUTPUT_MERGED/PANDAS'
data = pd.read_csv(f"{PANDAS_DIR}/{RUN}.csv")
sub_id = list(data['sub_id'])
event_id = list(data['event_id'])
task_id = [f"{x}_{y}" for x, y in zip(sub_id, event_id)]
task_id_np = np.asarray(task_id)

# PROPS IS THE NUMPY ARRAY (N_TASKID, 20) [TEST.CSV HAS 140 TASK-ID WITH 20 PROBABILITIES OF 20 ACTIVITIES]
#probs = joblib.load(f'joblib_files/autographer_prediction_test_RUN_6_Unfreeze.joblib')
probs = joblib.load(f'joblib_files/tabular_r34_prediction.joblib')

# GENERATE TXT FILE
submission = "group_id: group14 forests 4\n"

for act in range(20):
    probs_act = probs[:, act]
    highest_subj = np.zeros(7)
    for i in range(7):
        subj_act_probs = probs_act[(i*20):((i+1)*20)]
        highest_index = np.argsort(subj_act_probs)[::-1][0]
        highest_index += (i*20)
        highest_subj[i] = highest_index
    highest_subj = highest_subj.astype(int)
    highest_subj_list = list(highest_subj)
    remain_subj_act = [x for x in range(len(task_id)) if x not in highest_subj_list]
    remain_subj_act = np.asarray(remain_subj_act)
    highest_subj_score = probs_act[highest_subj]
    remain_subj_score = probs_act[remain_subj_act]
    
    h_index = np.argsort(highest_subj_score)[::-1]
    h_map_index = highest_subj[h_index]
    r_index = np.argsort(remain_subj_score)[::-1]
    r_map_index = remain_subj_act[r_index]
    final_index = np.concatenate((h_map_index, r_map_index))
    task_ranking = task_id_np[final_index]
    task_score = probs_act[final_index]
    
    task_ranking_list = list(task_ranking)
    if act < 9:
        act_str = f"act0{act+1}"
    else:
        act_str = f"act{act+1}"
    for task in task_ranking_list:
        submission += f"{act_str} {task}\n"

submission_write = open(f"Submission/submission_test_tabular_r34.txt", "w")
submission_write.write(submission)
submission_write.close()

print('DONE')