# -*- coding: utf-8 -*-
from evaluate import strict, loose_macro, loose_micro

def get_true_and_prediction(scores, y_data):
    true_and_prediction = []
    for score,true_label in zip(scores,y_data):
        predicted_tag = []
        true_tag = []
        for label_id,label_score in enumerate(list(true_label)):
            if label_score > 0:
                true_tag.append(label_id)
        lid,ls = max(enumerate(list(score)),key=lambda x: x[1])
        predicted_tag.append(lid)
        for label_id,label_score in enumerate(list(score)):
            if label_score > 0.5:
                if label_id != lid:
                    predicted_tag.append(label_id)
        true_and_prediction.append((true_tag, predicted_tag))
    return true_and_prediction

def acc_hook(scores, y_data):
    true_and_prediction = get_true_and_prediction(scores, y_data)
    strict_ = strict(true_and_prediction)
    loose_macro_ = loose_macro(true_and_prediction)
    loose_micro_ = loose_micro(true_and_prediction)

    print("     strict (p,r,f1):", strict_)
    print("loose macro (p,r,f1):", loose_macro_)
    print("loose micro (p,r,f1):", loose_micro_)

    save_acc_hook(strict_, loose_macro_, loose_micro_)

def save_predictions(scores, y_data, id2label, fname):
    true_and_prediction = get_true_and_prediction(scores, y_data)
    with open(fname,"w") as f:
        for t, p in true_and_prediction:
            f.write(" ".join([id2label[id] for id in t]) + "\t" + " ".join([id2label[id] for id in p]) + "\n")
    f.close()

def save_acc_hook(strict_, loose_macro_, loose_micro_):
    with open('acc_hook_loop.log',"a+") as f:
        f.write("     strict (p,r,f1):" + str(strict_) + "\n")
        f.write("loose macro (p,r,f1):" + str(loose_macro_) + "\n")
        f.write("loose micro (p,r,f1):" + str(loose_micro_) + "\n\n")
    f.close()
    