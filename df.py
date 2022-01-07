import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn import preprocessing
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import roc_auc_score, roc_curve, auc

sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
df=pd.read_csv("D01_20190206_centralesupelec.csv")
df=df.dropna()
#print(df.shape)
#print(list(df.columns))
new=df['ident\tgrossesse\tallaitementoui\tageq3\talcool\tkcalsac\taphyq3\tATCDfamdiabQ8\tdiab\tdt_diab\tencoupleq1\tetude\tFDEP99\tFRAP_I_NC\tgpsang_rhe\thypertensionq3\thypolipi2\tttailleq4\timcq4\tlateralite\tstatmeno_q3_cl\tpattern_western\tpattern_prudent\tpoidsnaiss\tPRAL\tageregle\tsommeil\ttabacq3\ttransitq4'].str.split("\t", expand = True)
new.columns=['ident','grossesse','allaitementoui','ageq3','alcool','kcalsac','aphyq3','ATCDfamdiabQ8','diab','dt_diab','tencoupleq1','etude','FDEP99','FRAP_I_NC','gpsang_rhe','hypertensionq3','hypolipi2','ttailleq4','imcq4','lateralite','statmeno_q3_cl','pattern_western','pattern_prudent','poidsnaiss','PRAL','ageregle','sommeil','tabacq3','transitq4']
#plt.scatter(new['alcool'],new['diab'],marker='+',color='red')
# X_train, X_test, y_train, y_test = train_test_split(new[['ATCDfamdiabQ8']],new.diab,train_size=0.9)

del new['dt_diab']

# =============================================================================
# construction et modèle avec des X_app,X_test,y_app,y_test  normaux
# =============================================================================

#X_app,X_test,y_app,y_test = train_test_split(new.loc[:, new.columns != 'diab'], new['diab'],test_size = 10000,random_state=0)
#
##model = LogisticRegression(solver = 'newton-cg', multi_class = 'auto')
#model = LogisticRegression()
#
#le = preprocessing.LabelEncoder()
#for column_name in X_app.columns:
#    if X_app[column_name].dtype == object:
#        X_app[column_name] = le.fit_transform(X_app[column_name])
#    else:
#        pass
#for column_name in X_test.columns:
#    if X_test[column_name].dtype == object:
#        X_test[column_name] = le.fit_transform(X_test[column_name])
#    else:
#        pass
#    
#
#lr=model.fit(X_app,y_app)
#y_predicted = lr.predict_proba(X_test)


# =============================================================================
# construction et modèle avec des X_app,X_test,y_app,y_test en augmentant la proportion de malades
# =============================================================================

X_app,X_test,y_app,y_test = train_test_split(new.loc[:, new.columns != 'diab'], new['diab'],test_size = 10000,random_state=1)
nvappdiab=new[new['diab']=='1']
nvssdiab=new[new['diab']=='0']
newapp=pd.concat([nvappdiab.loc[1:72000, :], nvssdiab.loc[1:40000, :]])
ynewapp = newapp['diab']
del newapp['diab']

#model = LogisticRegression(solver = 'newton-cg', multi_class = 'auto')
model = LogisticRegression(C=0.0001)

le = preprocessing.LabelEncoder()
for column_name in newapp.columns:
    if newapp[column_name].dtype == object:
        newapp[column_name] = le.fit_transform(newapp[column_name])
    else:
        pass
for column_name in X_test.columns:
    if X_test[column_name].dtype == object:
        X_test[column_name] = le.fit_transform(X_test[column_name])
    else:
        pass
    

lr=model.fit(newapp,ynewapp)
y_predicted = lr.predict_proba(X_test)


#print(' ')
#print('*************  coefficients   ****************')
#print(' ')
#print(lr.coef_)
#print(' ')

# =============================================================================
# choix de la probabilite limite p au dessus de laquelle on prédit le patient malade
# =============================================================================

#p=0.4
#
#y_pred = np.array(y_predicted)
#y_predicted_plt=[]
#for i in range(len(y_predicted)):
#    if y_pred[i][1] < p:
#        y_predicted_plt.append(0)
#    else :
#        y_predicted_plt.append(1)
#        
#T= np.zeros(len(y_predicted))
#T=T+p        
#y_predicted_proba=[]
#for k in range(len(y_predicted)):
#    y_predicted_proba.append(y_predicted[k][1])
#
#plt.figure(1)
#plt.subplot(311)
#plt.plot(y_predicted_proba,'bs',T,'r--')
#plt.subplot(312)
#plt.plot(y_predicted_plt, 'rs')
#plt.subplot(313)
#plt.plot(y_test,'gs')
#plt.show()
#
#y_test=np.array(y_test)  
#matrice_confusion = np.array([[0,0],[0,0]])
#for j in range(len(y_predicted_plt)):
#    if y_predicted_plt[j]==1 and y_test[j]=='1':
#        matrice_confusion[1][1]=matrice_confusion[1][1]+1
#    if y_predicted_plt[j]==0 and y_test[j]=='1':
#        matrice_confusion[1][0]=matrice_confusion[1][0]+1
#    if y_predicted_plt[j]==1 and y_test[j]=='0':
#        matrice_confusion[0][1]=matrice_confusion[0][1]+1
#    if y_predicted_plt[j]==0 and y_test[j]=='0':
#        matrice_confusion[0][0]=matrice_confusion[0][0]+1
#
#
#total = matrice_confusion[1][1]+matrice_confusion[0][0]+matrice_confusion[0][1]+matrice_confusion[1][0]
#taux_correct = (matrice_confusion[0][0]+matrice_confusion[1][1])/total
#specificite = matrice_confusion[1][1]/(matrice_confusion[1][0]+matrice_confusion[1][1])
#sensibilite = matrice_confusion[0][0]/(matrice_confusion[0][0]+matrice_confusion[0][1])
#pos_vraie = matrice_confusion[1][1]/(matrice_confusion[0][1]+matrice_confusion[1][1])
#neg_vraie = matrice_confusion[0][0]/(matrice_confusion[0][0]+matrice_confusion[1][0])
#
#print('***************  matrice de confusion   ***************')
#print(' ')
#print(matrice_confusion)
#print(' ')
#print('taux_correct',taux_correct)
#print('specificite',specificite)
#print('sensibilite',sensibilite)
#print('pos_vraie',pos_vraie)
#print('neg_vraie',neg_vraie)


# =============================================================================
# test probabilité limite optimale au dessus de laquelle on prédit le patient malade
# =============================================================================

y_pred = np.array(y_predicted)
y_test_plt=np.array(y_test)
p_opt=0
p=0
indice = 0
best_cas =  np.array([[0,0],[0,0]])
y_predicted_plt_final=[]
taux_correct_final = 0
specificite_final = 0
sensibilité_final = 0
pos_vraie_final = 0
neg_vraie_final = 0

while p < 1 :
        
# construction de y_predicted final  
    y_predicted_plt=[]
    for i in range(len(y_predicted)):                                      
        if y_pred[i][1] < p :
            y_predicted_plt.append(0)
        else :
            y_predicted_plt.append(1) 
 
# construction matrice de confusion
    matrice_confusion = np.array([[0,0],[0,0]])
    for j in range(len(y_predicted_plt)):                                  
        if y_predicted_plt[j]==1 and y_test_plt[j]=='1':
            matrice_confusion[1][1] = matrice_confusion[1][1]+1
        if y_predicted_plt[j]==0 and y_test_plt[j]=='1':
            matrice_confusion[1][0] = matrice_confusion[1][0]+1
        if y_predicted_plt[j]==1 and y_test_plt[j]=='0':
            matrice_confusion[0][1] = matrice_confusion[0][1]+1
        if y_predicted_plt[j]==0 and y_test_plt[j]=='0':
            matrice_confusion[0][0] = matrice_confusion[0][0]+1
    
    total = matrice_confusion[1][1]+matrice_confusion[0][0]+matrice_confusion[0][1]+matrice_confusion[1][0] 
    taux_correct = (matrice_confusion[0][0]+matrice_confusion[1][1])/total
    specificite = matrice_confusion[1][1]/(matrice_confusion[1][0]+matrice_confusion[1][1])
    sensibilite = matrice_confusion[0][0]/(matrice_confusion[0][0]+matrice_confusion[0][1])
    pos_vraie = matrice_confusion[1][1]/(matrice_confusion[0][1]+matrice_confusion[1][1])
    neg_vraie = matrice_confusion[0][0]/(matrice_confusion[0][0]+matrice_confusion[1][0])
    i = 0.2*taux_correct + 0.2*specificite + 0.2*sensibilite + 0.2*pos_vraie + 0.2*neg_vraie
    if i > indice  and specificite > 0.5:
        p_opt = p
        indice = i
        best_cas = matrice_confusion
        y_predicted_plt_final = y_predicted_plt
        taux_correct_final = taux_correct
        specificite_final = specificite
        sensibilite_final = sensibilite
        pos_vraie_final = pos_vraie
        neg_vraie_final = neg_vraie
    p = p + 0.01

#print('***************  y_predicted  *************')
#print(' ')
#print(y_predicted_plt_final)
#print(' ')        
print('***************  matrice de confusion   ***************')
print(' ')
print(best_cas)
print(' ')
print('p = ', p_opt)
print('taux_correct_final',taux_correct_final)
print('specificite_final',specificite_final)
print('sensibilite_final',sensibilite_final)
print('pos_vraie_final',pos_vraie_final)
print('neg_vraie_final',neg_vraie_final)
print('indice',indice)

T= np.zeros(len(y_predicted))
T=T+p_opt
y_predicted_proba=[]
for k in range(len(y_predicted)):
    y_predicted_proba.append(y_predicted[k][1])

plt.figure(1)
plt.subplot(311)
plt.plot(y_predicted_proba,'bs',T,'r--')
plt.subplot(312)
plt.plot(y_predicted_plt_final, 'rs')
plt.subplot(313)
plt.plot(y_test,'gs')
plt.show()

# =============================================================================
sns.countplot(x='diab',data=new, palette='hls')
sns.heatmap(new.corr())
from sklearn.metrics import classification_report
print(classification_report(y_test, y_predicted))
#l'influence de l'hypertention  et antécédents de diabète dans la famille sur le résultat 
g = sns.FacetGrid(new, col='diab')
g.map(plt.hist, 'ATCDfamdiabQ8', bins=20)
g = sns.FacetGrid(new, col='diab')
g.map(plt.hist, 'hypertensionq3', bins=3)
#l'histogramme de toutes les variables
new.groupby('diab').hist(figsize=(27, 27))
# =============================================================================
# la fonction de ROc
# =============================================================================
from sklearn.metrics import roc_auc_score, roc_curve, auc
probas = lr.predict_proba(X_test)
fpr0, tpr0, thresholds0 = roc_curve(y_test, probas[:, 0], pos_label=lr.classes_[0], drop_intermediate=False)
fpr0.shape
fig, ax = plt.subplots(1, 1, figsize=(4,4))
ax.plot([0, 1], [0, 1], 'k--')
# aucf = roc_auc_score(y_test == clr.classes_[0], probas[:, 0]) # première façon
aucf = auc(fpr0, tpr0)  # seconde façon
ax.plot(fpr0, tpr0, label=lr.classes_[0] + ' auc=%1.5f' % aucf)
ax.set_title('Courbe ROC - classifieur couleur des vins')
ax.text(0.5, 0.3, "plus mauvais que\nle hasard dans\ncette zone")
ax.legend();
#==================================================================

# matrice de confusion
# =============================================================================

matrice_confusion = confusion_matrix(y_test, y_predicted)
print(matrice_confusion)
# =============================================================================
from sklearn.feature_selection import RFECV
# Create the RFE object and compute a cross-validated score.
# The "accuracy" scoring is proportional to the number of correct classifications
rfecv = RFECV(estimator=LogisticRegression(), step=1, cv=10, scoring='accuracy')
rfecv.fit(X_app, y_app)
print("Optimal number of features: %d" % rfecv.n_features_)
print('Selected features: %s' % list(X_app.columns[rfecv.support_]))

# Plot number of features VS. cross-validation scores
plt.figure(figsize=(10,6))
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()
# =============================================================================
# random forest
# =============================================================================
# Create a random Forest Classifier instance
rfc = RandomForestClassifier(n_estimators = 100)
# Fit to the training data
rfc.fit(X_train, y_train)
# Predict on the test data: y_pred
y_pred = rfc.predict(X_test)

# Score / Metrics
accuracy = rfc.score(X_test, y_test) # = accuracy
MlResult('Random Forest',accuracy)
# =============================================================================
# exporter le rapport 
# =============================================================================
from jinja2 import Environment, FileSystemLoader
env = Environment(loader=FileSystemLoader('.'))
template = env.get_template('C:/Users/E330/Desktop/projet supelec/rapp.html')
template_vars = {"title" : "diabet",
                 "national_pivot_table":df.to_html()}
html_out = template.render(template_vars)
# =============================================================================
# la fonction auc
# =============================================================================
y_proba = lr.predict_proba(X_test)
from sklearn.metrics import log_loss
err = log_loss(y_test, y_proba)
print(err)
y_probal = lrl.predict_proba(X_test)
from sklearn.metrics import log_loss
errl = log_loss(y_test, y_probal)
print(errl)
# =============================================================================
# courbe de ROC
# =============================================================================
rom sklearn.metrics import roc_auc_score, roc_curve, auc
probas = lr.predict_proba(X_test)
fpr0, tpr0, thresholds0 = roc_curve(y_test, probas[:, 0], pos_label=lr.classes_[0], drop_intermediate=False)
fpr0.shape
fig, ax = plt.subplots(1, 1, figsize=(4,4))
ax.plot([0, 1], [0, 1], 'k--')

# =============================================================================
# fonction sigmoid
# =============================================================================
#
def model(lr,x):
  return sigmoid(lr.coef_.T[7]*x['ATCDfamdiabQ8']+lr.coef_.T[13]*x['hypertensionq3']+lr.coef_.T[15]*x['ttailleq4']+lr.coef_.T[14]*x['hypolipi2']+lr.intercept_)
#
for i in range(X_test.shape[0]):
    plot(X_test[['ttailleq4']].iloc[i,:],model(lr,X_test[['ATCDfamdiabQ8','hypertensionq3','hypolipi2','ttailleq4']].iloc[i,:]),'ro',color='green')
plot(X_test['ttailleq4'],y_predicted,'ro')
def sigmoid(t):                          # construction de sigmoid 
    return (1/(1 + np.e**(-t)))

def model(lr,x,i):
    return sigmoid(np.dot(X_test.iloc[i,:],np.transpose(lr.coef_))+lr.intercept_)

for i in range(X_test.shape[0]):
    plot(i,model(lrl,X_test,i),'-o',color='green')
plot(X_test['ttailleq4'],y_predicted,'ro')
plt.x=x
plt.y=y 
plt.plot(X_test[['ttailleq4']], lrl.predict_proba(X_test)[:,0], '.', label='Logistic regr')
plt.plot(X_test[['ttailleq4']], lrl.predict_proba(X_test)[:,1], '.', label='Logistic regr') 
# =============================================================================
# les p values et les intervalles de confiance
# =============================================================================
import statsmodels.formula.api as sm
 
model = sm.Logit(s.astype(int),new.astype(int))
 
result = model.fit()
result.summary

