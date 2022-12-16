import seaborn as sns
from sklearn.metrics import confusion_matrix,f1_score,recall_score,precision_score,balanced_accuracy_score
from sklearn.metrics import plot_confusion_matrix,f1_score, accuracy_score, average_precision_score

def eval_all(predict_list, truth_label):
    #creator
    acc=0
    for index in range(len(predict_list[0])):
      acc = acc + accuracy_score(truth_label[0][index], predict_list[0][index])
    acc = acc / len(predict_list[0])
    f1_c=f1_score(truth_label[0],predict_list[0],average='micro')
    recal_c = recall_score(truth_label[0],predict_list[0],average='micro')
    prec_c = precision_score(truth_label[0],predict_list[0],average='micro')
    mac_c = balanced_accuracy_score(truth_label[0],predict_list[0])
    con_mat_c=confusion_matrix(truth_label[0],predict_list[0])

    #material
    mmAp = 0
    for index in range(len(predict_list[1])):
      mmAp = mmAp + average_precision_score(truth_label[1][index], predict_list[1][index], average='weighted')
    mmAp = mmAp / len(predict_list[1])

    mmRecall = 0
    for index in range(len(predict_list[1])):
      mmRecall = mmRecall + recall_score(truth_label[1][index], predict_list[1][index], average='weighted')
    mmRecall = mmRecall/len(predict_list[1])

    mf1 = 2*(mmAp*mmRecall)/(mmAp+mmRecall)

    #type
    tmAp = 0
    for index in range(len(predict_list[2])):
      tmAp = tmAp + average_precision_score(truth_label[2][index], predict_list[2][index], average='weighted')
    tmAp = tmAp / len(predict_list[2])

    tmRecall = 0
    for index in range(len(predict_list[2])):
      tmRecall = tmRecall + recall_score(truth_label[2][index], predict_list[2][index], average='weighted')
    tmRecall = tmRecall/len(predict_list[2])

    tf1 = 2*(tmAp*tmRecall)/(tmAp+tmRecall)

    return [f1_c,recal_c,prec_c,prec_c,mac_c,con_mat_c,acc], [mmAp,mmRecall,mf1], [tmAp,tmRecall,tf1]
