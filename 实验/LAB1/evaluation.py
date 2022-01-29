import numpy as np
# 得到y中所有元素的集合
def total_label(y):
    y=y.reshape(y.shape[0],).tolist()
    return set(y)


# 根据label，将原来的多分类任务视为二分类任务，得到对应的TP,FP,FN
def get_binary_TP_FP_FN(y,pred,label):
    num=y.shape[0]
    # pred=pred.astype(int).tolist()
    TP=0
    FP=0
    FN=0
    TN=0
    for i in range(num):  
        if y[i][0]==label and pred[i][0]==label:
            TP+=1
        elif y[i][0]==label and pred[i][0]!=label:
            FN+=1
        elif y[i][0]!=label and pred[i][0]==label:
            FP+=1
        else:
            TN+=1
    return TP,FP,FN

# 根据TP,FP,FN计算得到P,R
def get_binary_P_R(TP,FP,FN):
    return TP/(TP+FP),TP/(TP+FN)

# 根据P,R计算得到对应的F1-score
def get_binary_f1(P,R):
    return 2*P*R/(P+R)

# 计算准确率
def get_acc(y,pred):
    return np.sum(y==pred)/len(y)

# 计算二分类的F1
def get_F1(y,pred,label):
    TP,FP,FN = get_binary_TP_FP_FN(y,pred,label)
    P,R = get_binary_P_R(TP,FP,FN)
    F1 = get_binary_f1(P,R)
    return P,R,F1


# 计算多分类的macro-F1
def get_macro_F1(y,pred):
    label_set=total_label(y)
    F1_score=0
    for label in label_set:
        TP,FP,FN=get_binary_TP_FP_FN(y,pred,label)
        P,R=get_binary_P_R(TP,FP,FN)
        score=get_binary_f1(P,R)
        print(score)
        F1_score+=score
    return F1_score/(len(label_set))

# 计算多分类的micro-F1
def get_micro_F1(y,pred):
    label_set=total_label(y)
    F1_score=0
    total_TP=0
    total_FP=0
    total_FN=0
    for label in label_set:
        TP,FP,FN=get_binary_TP_FP_FN(y,pred,label)
        total_TP+=TP
        total_FP+=FP
        total_FN+=FN
    P,R=get_binary_P_R(total_TP,total_FP,total_FN)
    F1_score+=get_binary_f1(P,R)    
    return F1_score
