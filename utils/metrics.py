from sklearn.metrics import confusion_matrix,plot_confusion_matrix,f1_score, roc_curve, auc,roc_auc_score, classification_report
from sklearn.metrics import precision_recall_curve

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def evaluate(model,X_test,y_test):
    """
    Evaluate test on different performances metrics
    
    """
    pred_proba = model.predict_proba(X_test)
    predictions = model.predict(X_test)
    print("==========CONFUSION MATRIX==========")
    # plot_confusion_matrix function is used to visualize the confusion matrix
    plot_confusion_matrix(model, X_test, y_test)
    plt.show()
    
    # Classification report
    print("==========CLASSIFICATION REPORT==========")
    print(classification_report(y_test, predictions))

    
    
    print("==========AUC/ROC CURVE==========")
    class_probabilities = pred_proba 
    preds = class_probabilities[:, 1]

    fpr, tpr, threshold = roc_curve(y_test, preds)
    roc_auc = auc(fpr, tpr)

    # Printing AUC
    print(f"AUC for our classifier is: {roc_auc}")

    # Plotting the ROC
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    
    print("==========PR CURVE==========")
    # predict probabilities

    lr_precision, lr_recall, _ = precision_recall_curve(y_test, preds)

    lr_f1, lr_auc = f1_score(y_test, predictions), auc(lr_recall, lr_precision)
    # summarize scores
    print('Logistic: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))
    # plot the precision-recall curves
    no_skill = len(y_test[y_test==1]) / len(y_test)
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    plt.plot(lr_recall, lr_precision, marker='.', label='Logistic')
    # axis labels
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    # show the legend
    plt.legend()
    # show the plot
    plt.title("Model's PR curve")
    plt.show()
    
def evaluate_test(class_probabilities,y_test):
    """
    returns auc, pr  and roc curve from Lunit prediction
    
    """
    
    print("==========AUC/ROC CURVE==========")
    preds = class_probabilities[:, 1]

    fpr, tpr, threshold = roc_curve(y_test, preds)
    roc_auc = auc(fpr, tpr)

    # Printing AUC
    print(f"AUC for our classifier is: {roc_auc}")

    # Plotting the ROC
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    
    print("==========PR CURVE==========")
    # predict probabilities

    lr_precision, lr_recall, _ = precision_recall_curve(y_test, preds)

    lr_auc =  auc(lr_recall, lr_precision)
    # summarize scores
    print('Logistic: auc=%.3f' % (lr_auc))
    # plot the precision-recall curves
    no_skill = len(y_test[y_test==1]) / len(y_test)
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    plt.plot(lr_recall, lr_precision, marker='.', label='Logistic')
    # axis labels
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    # show the legend
    plt.legend()
    # show the plot
    plt.title("Model's PR curve")
    plt.show()
   