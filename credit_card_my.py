#-*- coding: utf8 -*-
"""
@author: sophiatree

与credit_card_my.ipynb配套使用，方便模型测试
"""

import numpy as np
import pandas as pd
import seaborn as sns
import time,os,copy,warnings
import copy

from collections import Counter
from math import log
import scorecard_function

import matplotlib.pyplot as plt
%matplotlib inline 

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import Binarizer

from imblearn.over_sampling import SMOTE

from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.stats import kendalltau
from scipy.stats import mode

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFE

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import ShuffleSplit

from sklearn import metrics
import pydotplus
from IPython.display import Image 

pd.set_option('display.precision', 3)
warnings.filterwarnings("ignore")

#模型评价函数
def plot_confusion_matrix(conf_mat):
    with sns.axes_style("darkgrid"):
        fig, ax = plt.subplots(figsize=(2.5, 2.5))
        ax.matshow(conf_mat,cmap=plt.cm.Blues,alpha=0.3)
        for i in range(conf_mat.shape[0]):
            for j in range(conf_mat.shape[1]):
                ax.text(x=j, y=i,
                        s=conf_mat[i, j],
                        va='center', ha='center')
        plt.xlabel('predicted label')
        plt.ylabel('true label')
        plt.show()

def plot_roc_curve(test_y,predicted_proba):
    with sns.axes_style("darkgrid"):
        fpr,tpr,thresholds = metrics.roc_curve(test_y, predicted_proba)
        roc_auc = metrics.auc(fpr,tpr)
        print roc_auc
        plt.plot(fpr,tpr,lw=1,label='AUC = %.3f' % roc_auc)
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('Receiver Operating Characteristic Curve')
        plt.plot(fpr,fpr,ls='--')
        plt.legend()
        plt.show()

def plot_ks_curve(test_y,predicted_proba):
    with sns.axes_style("darkgrid"):
        fpr,tpr,thresholds = metrics.roc_curve(test_y, predicted_proba)
        ks_x = np.linspace(0,1,fpr.shape[0])
        ks_value = np.max(tpr-fpr)
        print ks_value
        ks_max_posit = np.argmax(tpr-fpr)
        print 'ks_max_threshold'
        print thresholds[ks_max_posit]
        plt.plot(ks_x,tpr,label='TPR')
        plt.plot(ks_x,fpr,label='FPR')
        plt.legend()
        plt.axvline(x=ks_x[ks_max_posit],ymin=fpr[ks_max_posit]+0.03,ymax=tpr[ks_max_posit]-0.03,ls=':')
        plt.text(x=ks_x[ks_max_posit]+0.02,y=(fpr[ks_max_posit]+tpr[ks_max_posit])/2.0,s='K-S value=%.3f' % ks_value)
        plt.xlabel('Sample proportion')
        plt.ylabel('TPR/FPR')
        plt.title('K-S curve')
        plt.legend()
        plt.show()

#等宽
def psi(test1,test2,bins):
    with sns.axes_style("darkgrid"):
        cat1 = pd.cut(test1,bins = bins,include_lowest=True)
        cat2 = pd.cut(test2,bins = bins,include_lowest=True)#include_lowest=True必须有
        a = pd.value_counts(cat1).sort_index()
        b = pd.value_counts(cat2).sort_index()

        #平滑
        a[a==0]=1
        b[b==0]=1

        a = a/sum(a)
        b = b/sum(b)

        from math import log
        #避免x为0,需要平滑，这里需要说明的是，不能直接将x平滑为0，因为log(0)为-∞，它减去负数应该还为负数，但是若平滑为0，0减去负数为正数，这会导致最终psi可能为负数
        #需在前一步就频数平滑+1
        log_a = a.apply(lambda x:log(x))
        log_b = b.apply(lambda x:log(x))
        return sum((b-a)*(log_b-log_a))

#等深
def psi_q(test1,test2,n):
    with sns.axes_style("darkgrid"):
        cat1 = pd.qcut(test1,n,retbins=True)
        #由于数据分布原因：前段数据过多，等分出现了重复bin，此时会报错，测试发现最大只能4等分。因而这里输入n=4
        #另外，即使调小能得到结果，也不一定是真正的4等分，因为unique bin并不代表里面的数据是unique的。
        cat2 = pd.cut(test2,bins = cat1[1],retbins=True,include_lowest=True)#include_lowest=True必须有
        a = pd.value_counts(cat1[0]).sort_index()
        b = pd.value_counts(cat2[0]).sort_index()

        #平滑
        a[a==0]=1
        b[b==0]=1

        a = a/sum(a)
        b = b/sum(b)

        from math import log
        log_a = a.apply(lambda x:log(x))
        log_b = b.apply(lambda x:log(x))
        return sum((b-a)*(log_b-log_a))

def plot_learning_curve(model,X_train,y_train,train_sizes,scoring):
    with sns.axes_style("darkgrid"):
        train_sizes, train_scores, test_scores = learning_curve(model, X_train, y_train,
                                                                train_sizes=train_sizes,scoring=scoring)
        train_mean = train_scores.mean(axis=1)
        train_std = train_scores.std(axis=1)
        test_mean = test_scores.mean(axis=1)
        test_std = test_scores.std(axis=1)
        plt.plot(train_sizes, train_mean,
            color='blue', marker='o',
            markersize=5,
            label='training AUC')
        plt.fill_between(train_sizes,
            train_mean + train_std,
            train_mean - train_std,
            alpha=0.15, color='blue')
        plt.plot(train_sizes, test_mean,
            color='green', linestyle='--',
            marker='s', markersize=5,
            label='validation AUC')
        plt.fill_between(train_sizes,
            test_mean + test_std,
            test_mean - test_std,
            alpha=0.15, color='green')
        plt.xlabel('Number of training samples')
        plt.ylabel('ROC_AUC')
        plt.legend(loc='lower right')


if __name__ == '__main__':
    pwd = os.getcwd()
    data_file = pwd+"/credit_card_data.csv"
    data_s = pd.read_csv(data_file)

    #--------------------------------  数据准备  --------------------------------

    data = data_s.drop('Unnamed: 0',axis=1)

    #重命名特征名
    original_columns = data.columns.values
    new_columns = ['y','x1','x2','x3','x4','x5','x6','x7','x8','x9','x10']
    data.columns=[new_columns]
    dict_col = {}
    for i in range(len(original_columns)):
        dict_col[new_columns[i]] = original_columns[i]


    #划分训练集（包括验证集）+测试集
    y = data['y'].values
    x = data.drop('y',axis=1).values
    ##分层抽样,n_splits=5是为了提取0.2比例的数据，但我们并不需要全部的5折，因为我们是为了分层抽样划分测试集，而不是为了交叉验证
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=5,random_state=1,shuffle=True)
    train_index = next(skf.split(x,y))[0]
    test_index = next(skf.split(x,y))[1]
    ## 注意：index 仍然保留着data中的index（行序号不变）
    train = data.iloc[train_index,:]
    test = data.iloc[test_index,:] 


    #--------------------------------  数据清洗  --------------------------------

    #样本是否均衡
    print train['y'].value_counts()
    sns.countplot(train['y'])

    #重复值
    train.drop_duplicates(inplace = True)

    #缺失值
    """
    train.x5.fillna(train.x5.median(),inplace=True)# MonthlyIncome，中位数填充
    from scipy.stats import mode
    train.x10.fillna(train.x10.mode()[0],inplace=True)# NumberOfDependents，众数填充
    """
    #可以用上面的注释部分进行处理，但是有个问题：没有封装成类，处理另一批数据（例如test数据集）时会产生错误，因而建议使用Imputer
    from sklearn.preprocessing import Imputer
    from sklearn.ensemble import RandomForestRegressor 

    ##x5
    def set_missing(data):
        known = data[data['x5'].notnull()]
        unknown = data[data['x5'].isnull()]

        X = known.drop(['x5','x10'],axis=1)#x10也有缺失值，代入模型会报错
        y = known['x5']
        clf = RandomForestRegressor().fit(X,y)
        predicted = clf.predict(unknown.drop(['x5','x10'],axis=1))
        return clf,predicted

    imp5,predicted = set_missing(train)
    train.loc[train['x5'].isnull(),'x5'] = predicted

    ##x10
    imp10 = Imputer(missing_values='NaN',strategy='most_frequent')
    imp10.fit(train.iloc[:,10:11])
    train.x10 = imp10.transform(train.iloc[:,10:11])


    #异常值
    train.x1[train.x1>1] = 1 
    train = train[train['x3']<90]
    train.x4[train.x4>1] = 1

    X = train.iloc[:,1:]
    y = train.iloc[:,0]
    m,n = train.shape

    #分箱离散化

    ##等宽分箱（基于数据的理解人为分箱）
    x1_bin = list(np.linspace(0.1,0.9,num=9))
    x2_bin = range(25,105,5)
    x3_bin = range(1,11)
    x7_bin = range(1,11)
    x9_bin = range(1,11)
    x4_bin = list(np.linspace(0.1,0.9,num=9))
    x5_bin = range(1000,14000,1000)+range(20000,50000,10000)
    x6_bin = range(1,20)
    x8_bin = range(1,10)
    x10_bin = range(1,5)


    ## 卡方分箱
    
    print '卡方分箱结果为：'
    for x in X.columns:
        locals()[x+'_chi'] = scorecard_function.ChiMerge(train, x, 'y', max_interval=10, special_attribute=[], minBinPcnt=0) 
        print x , locals()[x+'_chi']   

    ##决策树分箱
    from sklearn import tree
    def decession_tree_bin(X,y,bin):
        clf = tree.DecisionTreeClassifier(criterion = 'entropy',max_leaf_nodes= bin,min_samples_leaf = 0.05).fit(X,y)
        n_nodes = clf.tree_.node_count
        children_left = clf.tree_.children_left
        children_right = clf.tree_.children_right
        threshold = clf.tree_.threshold
        boundary = []
        for i in range(n_nodes):
            if children_left[i]!=children_right[i]:
                boundary.append(threshold[i])
        sort_boundary = sorted(boundary)
        return sort_boundary

    print '决策树分箱结果为：'
    for x in X.columns:
        locals()[x+'_tre'] = decession_tree_bin( X[x].reshape(-1,1),y,bin=6)
        print x , locals()[x+'_tre'] 

    #下面三种取值方式都以卡方分箱xi_chi为例

    ##分箱内数值用分箱的均值/中位数/较近边界值代替
    ##需要注意的是，若用均值、中位数，上下边界需要指定，但每列的上下边界都不一样，为简便，这里采用较近边界值代替（取值个数只有：分箱数-1）
    def close(x,bins):
        t = [abs(x-i) for i in bins]
        p = t.index(min(t))
        return bins[p]

    for x in X.columns:
        locals()[x+'_me'] = copy.deepcopy(X[x])
        locals()[x+'_me'] = locals()[x+'_me'].map(lambda e:close(e,globals()[x+'_chi'])) #lambda函数内必须用全局声明才能访问函数外的变量
    X_me = pd.concat([locals()[x+'_me'] for x in X.columns],axis = 1)

    ##或者将每个分箱视为离散化后的类别（针对连续值，若是类别性数据还是需要LabelEncoder）
    for x in X.columns:
        locals()[x+'_le'] = X[x].map(lambda e:scorecard_function.AssignBin(e, globals()[x+'_chi'], special_attribute=[]))
    X_le = pd.concat([locals()[x+'_le'] for x in X.columns],axis = 1)

    ##用WOE编码，这是一种有监督的取值方式，通常用于信用评分卡
    for x in X.columns:
        locals()['woe_'+x] = scorecard_function.CalcWOE(pd.concat([X_le,y],axis=1), x, 'y')['WOE'] 
        locals()[x+'_woe'] = X_le[x].map(lambda e:globals()['woe_'+x][e]) #通过xi_le转变
    X_woe = pd.concat([locals()[x+'_woe'] for x in X.columns],axis = 1)



    #标准化
    from sklearn.preprocessing import StandardScaler
    s = StandardScaler().fit(X)
    X_s = pd.DataFrame(s.transform(X),columns = X.columns)
    
    me_s = StandardScaler().fit(X_me)
    X_me_s = pd.DataFrame(me_s.transform(X_me),columns = X_me.columns)


    #重采样
    ##X重采样
    from imblearn.over_sampling import SMOTE
    from collections import Counter

    ros = SMOTE(random_state=0,ratio = 'all',kind='borderline1')
    X_resampled, y_resampled = ros.fit_sample(X, y)
    sorted(Counter(y_resampled).items())
    train_resampled = np.hstack((X_resampled, y_resampled.reshape(-1,1)))
    np.random.shuffle(train_resampled)
    X_resampled = train_resampled[:,:-1]
    y_resampled = train_resampled[:,-1]

    X_resampled = pd.DataFrame(X_resampled ,columns=X.columns)
    y_resampled = pd.Series(y_resampled)

    ##X_s重采样
    ros = SMOTE(random_state=0,ratio = 'all',kind='borderline1')
    X_s_resampled, y_s_resampled = ros.fit_sample(X_s, y)
    sorted(Counter(y_s_resampled).items())

    train_s_resampled = np.hstack((X_s_resampled, y_s_resampled.reshape(-1,1)))
    np.random.shuffle(train_s_resampled)
    X_s_resampled = train_s_resampled[:,:-1]
    y_s_resampled = train_s_resampled[:,-1]

    X_s_resampled = pd.DataFrame(X_s_resampled,columns=X_s.columns)
    y_s_resampled = pd.Series(y_s_resampled)

    ##X,y,X_le,X_me,X_s,X_resampled, y_resampled,X_s_resampled,y_s_resampled已准备好

    #--------------------------------  特征工程  --------------------------------

    ## 这里需要大量的可视化的数据分析工作，推荐在jupyter notebook上操作
    ## 这里的train已经是经过初步数据清洗的数据

    """
    #首先大体观察各维度之间的两两关系，dropna=True的功能貌似没有完全实现，最好是在缺失值处理之后再画该图
    sns.pairplot(data = train.sample(frac=0.8),vars = ['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10'], hue="y",dropna=True)
    #具体观察两维度之间的关系
    sns.jointplot("x6", "x8", data=train.sample(frac=0.8), kind="reg")

    #Filter
    from sklearn.feature_selection import SelectPercentile
    from sklearn.feature_selection import SelectKBest

    ##变异系数，典型值是0.15，越大越好
    X_std = X.std()
    X_mean = X.mean()
    C_V = X_std.div(X_mean)
    print C_V

    ##pearsonr，针对连续定量变量，线性关系
    pearsonr_cor = train.corr('pearson') #并没有返回p值，需要另外计算p值并画图
    plt.figure(figsize=(10,10))
    sns.heatmap(pearsonr_cor, vmax=1, square=True, cmap="Blues",annot=True,fmt='.2f')

    from scipy.stats import pearsonr
    pearsonr_c = pd.DataFrame(np.arange(n*n).reshape((n,n)),index=train.columns,columns=train.columns)
    pearsonr_p = copy.deepcopy(pearsonr_c)

    for i in range(n):
        for j in range(n):
            pearsonr_c.iloc[i,j] = pearsonr(train.iloc[:,i],train.iloc[:,j])[0]
            pearsonr_p.iloc[i,j] = pearsonr(train.iloc[:,i],train.iloc[:,j])[1]

    plt.figure(figsize=(10,10))
    sns.heatmap(pearsonr_p, vmax=1, square=True, cmap="Blues",annot=True,fmt='.2f')#显著性检验，p值范围是[0,1]，值越小越好，即：热力图颜色越浅越好

    ##spearman,针对连续定量变量/定序变量，前提假设比pearsonr弱，只考虑有序等级
    from scipy.stats import spearmanr
    n = train.shape[1]
    spearmanr_c = pd.DataFrame(np.arange(n*n).reshape((n,n)),index=train.columns,columns=train.columns)
    spearmanr_p = copy.deepcopy(spearmanr_c)

    for i in range(n):
        for j in range(n):
            spearmanr_c.iloc[i,j] = spearmanr(train.iloc[:,i],train.iloc[:,j])[0]
            spearmanr_p.iloc[i,j] = spearmanr(train.iloc[:,i],train.iloc[:,j])[1]

    plt.figure(figsize=(10,10))
    sns.heatmap(spearmanr_p, vmax=1, square=True, cmap="Blues",annot=True,fmt='.2f')

    ##kendall秩相关系数，针对有序分类变量（定序变量）
    from scipy.stats import kendalltau
    n = train.shape[1]
    kendalltau_c = pd.DataFrame(np.arange(n*n).reshape((n,n)),index=train.columns,columns=train.columns)
    kendalltau_p = copy.deepcopy(kendalltau_c)

    for i in range(n):
        for j in range(n):
            kendalltau_c.iloc[i,j] = kendalltau(train.iloc[:,i],train.iloc[:,j])[0]
            kendalltau_p.iloc[i,j] = kendalltau(train.iloc[:,i],train.iloc[:,j])[1]

    plt.figure(figsize=(10,10))
    sns.heatmap(kendalltau_p, vmax=1, square=True, cmap="Blues",annot=True,fmt='.2f')

    ##互信息
    from sklearn.feature_selection import mutual_info_classif
    fs_m = SelectPercentile(mutual_info_classif,percentile=50).fit(X,y)
    fs_m.scores_

    
    ##MIC，经典的互信息是针对定性资料的，为了处理定量数据，最大信息系数法MIC被提出，计算速度慢。
    from minepy import MINE
    mine = MINE()
    mine_c = pd.DataFrame(np.arange(n*n).reshape((n,n)),index=train.columns,columns=train.columns)

    for i in range(n):
        for j in range(n):
            t = mine.compute_score(train.iloc[:,i],train.iloc[:,j])
            if t == None:
                mine_c.iloc[i,j] = -999
            else:
                mine_c.iloc[i,j] = t.mic()

    ##卡方检验，经典的卡方检验也是针对定性资料，但貌似sklearn包中已做了连续特征离散化操作，可以直接拿来用
    from sklearn.feature_selection import chi2
    fs_chi2 = SelectPercentile(chi2,percentile=50).fit(X,y)
    fs_chi2.scores_
    fs_chi2.pvalues_

    ##f检验
    from sklearn.feature_selection import f_classif
    fs_f = SelectPercentile(f_classif,percentile=50).fit(X,y)
    fs_f.scores_
    fs_f.pvalues_

    ##IV
    iv = []
    for i in X.columns:
        #print locals()[i+'_chi']
        iv.append(woe(train,i,locals()[i+'_chi'],'y').iloc[-1,-1])
    iv

    ##某种模型的交叉验证后平均score
    from sklearn.model_selection import cross_val_score
    from sklearn.ensemble import RandomForestClassifier

    RF = RandomForestClassifier(class_weight="balanced")#考虑样本不均衡问题
    scores=[]

    #对于DataFrame X，X.iloc[:, i:i+1]返回的是dataframe，shape为(m,1)；X.iloc[:,i]返回series，shape为(m,)。对于array X，同理
    for i in range(n-1):#n包括y
        score = cross_val_score(RF, X.iloc[:, i:i+1], y, cv=5,scoring="roc_auc")#考虑样本不均衡问题
        scores.append((X.columns[i],format(score.mean(),'.3f')))
    print sorted(scores,key=lambda e:e[1],reverse=True)#默认是从小到大的正序

    #Wrapper，主要指：递归消除特征RFE
    from sklearn.feature_selection import RFE
    from sklearn.linear_model import LogisticRegression
    LR = LogisticRegression(class_weight="balanced")

    rfe = RFE(estimator=LR, n_features_to_select=1).fit(X,y)
    #n_features_to_select=1是为了获得具体的排名，RFE会将挑选出来的特征都排名为1
    print rfe.ranking_

    #Embedded

    ##基于惩罚项
    from sklearn.feature_selection import SelectFromModel
    LR = LogisticRegression(class_weight="balanced",penalty="l1",C=1)#C为正则化系数λ的倒数，越小正则化越强
    sfm1 = SelectFromModel(LR,threshold=1e-02)#通过调整threshold来筛选特征
    X_new = sfm1.fit_transform(X,y)#返回特征选择后的X'（ndarray类型）
    print X_new.shape

    ##基于树模型
    from sklearn.ensemble import GradientBoostingClassifier
    GBDT = GradientBoostingClassifier().fit(X,y)
    X_new = SelectFromModel(GBDT,prefit=True).transform(X)
    
    """

    #--------------------------------  测试集准备  --------------------------------
    #重复值
    test.drop_duplicates(inplace = True)

    #缺失值
    if test['x5'].isnull().any() == True:#有缺失值才填充，否则会报错
        unknown = test[test['x5'].isnull()]
        test.loc[test['x5'].isnull(),'x5'] = imp5.predict(unknown.drop(['x5','x10'],axis=1))
    test['x10'] = imp10.transform(test.iloc[:,10:11])

    #异常值
    test.x1[test.x1>1] = 1 
    test = test[test['x3']<90]
    test.x4[test.x4>1] = 1

    #初步处理过的测试集
    test_X = test.iloc[:,1:]
    test_y = test.iloc[:,0]

    #分箱离散化
    ##分箱方法结果有xi_chi/xi_tre
    ###边界值代替
    for x in test_X.columns:
        locals()['test_'+x+'_me'] = test_X[x].map(lambda e:close(e,globals()[x+'_chi']))
    test_X_me = pd.concat([locals()['test_'+x+'_me'] for x in test_X.columns],axis = 1)


    ###类别代替
    for x in test_X.columns:
        locals()['test_'+x+'_le'] = test_X[x].map(lambda e:scorecard_function.AssignBin(e, globals()[x+'_chi'], special_attribute=[]))
    test_X_le = pd.concat([locals()['test_'+x+'_le'] for x in test_X.columns],axis = 1)

    ###woe编码
    for x in X.columns:
        locals()['test_'+x+'_woe'] = test_X_le[x].map(lambda e:globals()['woe_'+x][e]) 
    test_X_woe = pd.concat([locals()['test_'+x+'_woe'] for x in X.columns],axis = 1)

    #标准化
    test_X_s = pd.DataFrame(s.transform(test_X),columns = test_X.columns)
    test_X_me_s = pd.DataFrame(me_s.transform(test_X_me),columns = test_X_me.columns)

    ##test_X,test_y,test_X_le,test_X_me,test_X_s,test_X_me_s已准备好

    #--------------------------------  模型训练  --------------------------------
    #未标准化样本以决策树为例

    from sklearn.model_selection import GridSearchCV

    ##基本样本
    from sklearn.tree import DecisionTreeClassifier
    dt = DecisionTreeClassifier(class_weight='balanced',max_depth=8,min_samples_leaf=134,min_samples_split=492)
    dt.fit(X,y)

    predicted = dt.predict(test_X)
    predicted_proba = dt.predict_proba(test_X)

    print '决策树：'

    #混淆矩阵
    print "confusion_matrix:"
    c_matrix = metrics.confusion_matrix(test_y, predicted)
    plot_confusion_matrix(c_matrix)

    #分类报告
    print "classification_report:"
    print metrics.classification_report(test_y, predicted)

    #ROC-AUC
    print "AUC:"
    plot_roc_curve(test_y,predicted_proba[:,1])

    #KS
    print "KS:"
    plot_ks_curve(test_y,predicted_proba[:,1])

    #PSI
    print "PSI:"
    test1 = dt.predict(X)
    test2 = predicted
    bins = [0,0.5,1]
    print psi(test1,test2,bins)

    #学习曲线
    print 'learning curve:'
    plot_learning_curve(dt,X,y,train_sizes=np.linspace(0.1,1,10),scoring='roc_auc')



    #重采样
    dt_r = DecisionTreeClassifier(max_depth=10,min_samples_split=310,min_samples_leaf=9)
    dt_r.fit(X_resampled,y_resampled)

    predicted = dt_r.predict(test_X)
    predicted_proba = dt_r.predict_proba(test_X)

    #混淆矩阵
    print "confusion_matrix:"
    c_matrix = metrics.confusion_matrix(test_y, predicted)
    plot_confusion_matrix(c_matrix)

    #分类报告
    print "classification_report:"
    print metrics.classification_report(test_y, predicted)

    #ROC、AUC
    print "AUC:"
    plot_roc_curve(test_y,predicted_proba[:,1])#将分类结果：1视为正例

    #KS
    print "KS:"
    plot_ks_curve(test_y,predicted_proba[:,1])#将分类结果：1视为正例

    #PSI
    print "PSI:"
    test1 = dt_r.predict_proba(X)#注意：这里还是应该对X预测，而不是X_resampled
    test2 = predicted
    bins = [0,0.5,1]
    print psi(test1,test2,bins)

    #学习曲线
    print 'learning curve:'
    plot_learning_curve(dt_r,X_resampled,y_resampled,train_sizes=np.linspace(0.1,1,10),scoring='roc_auc')


    #分箱
    dt_me = DecisionTreeClassifier(class_weight='balanced',max_depth=7,min_samples_split=482,min_samples_leaf=51)
    dt_me.fit(X_me,y)

    predicted = dt_me.predict(test_X_me)
    predicted_proba = dt_me.predict_proba(test_X_me)

    #混淆矩阵
    print "confusion_matrix:"
    c_matrix = metrics.confusion_matrix(test_y, predicted)
    plot_confusion_matrix(c_matrix)

    #分类报告
    print "classification_report:"
    print metrics.classification_report(test_y, predicted)

    #ROC、AUC
    print "AUC:"
    plot_roc_curve(test_y,predicted_proba[:,1])#将分类结果：1视为正例

    #KS
    print "KS:"
    plot_ks_curve(test_y,predicted_proba[:,1])#将分类结果：1视为正例

    #PSI
    print "PSI:"
    test1 = dt_me.predict(X_me)
    test2 = predicted
    bins = [0,0.5,1]
    print psi(test1,test2,bins)

    #学习曲线
    print 'learning curve:'
    plot_learning_curve(dt_me,X_me,y,train_sizes=np.linspace(0.1,1,10),scoring='roc_auc')


    #模型解释

    ##1. 先生成dot文件，再在命令行生成pdf
    from sklearn import tree
    with open("dt.dot", 'w') as f:
        f = tree.export_graphviz(dt.best_estimator_, out_file=f,
                                feature_names=original_columns[1:], 
                            class_names=['Not','Yes'],
                              rounded=True)      
    #在终端对应路径下输入命令行：dot -Tpdf DT.dot -o DT.pdf

    ##2. pydotplus直接生成iris.pdf，避免切换到终端操作
    import pydotplus 
    dot_data = tree.export_graphviz(dt.best_estimator_, out_file=None,
                                   feature_names=original_columns[1:], 
                                    class_names=['Not','Yes'],
                                      rounded=True) 
    graph = pydotplus.graph_from_dot_data(dot_data) 
    graph.write_pdf("dt.pdf") 

    ##3. 在jupyter notebook上直接显示
    from IPython.display import Image  
    dot_data = tree.export_graphviz(dt.best_estimator_, out_file=None, 
                             feature_names=original_columns[1:], 
                            class_names=['Not','Yes'],
                              rounded=True)  
    graph = pydotplus.graph_from_dot_data(dot_data)  
    Image(graph.create_png())


    #标准化样本以逻辑回归为例
    #基本样本
    lr = LogisticRegression(penalty='l2',C=0.0002,class_weight='None')
    lr.fit(X_s,y)

    predicted = lr.predict(test_X_s)
    predicted_proba = lr.predict_proba(test_X_s)

    print 'lr:'

    print "confusion_matrix:"
    c_matrix = metrics.confusion_matrix(test_y, predicted)
    plot_confusion_matrix(c_matrix)

    print "classification_report:"
    print metrics.classification_report(test_y, predicted)

    print "AUC:"
    plot_roc_curve(test_y,predicted_proba[:,1])

    print "KS:"
    plot_ks_curve(test_y,predicted_proba[:,1])

    print "PSI:"
    test1 = lr.predict(X_s)
    test2 = predicted
    bins = [0,0.5,1]
    print psi(test1,test2,bins)

    print 'learning curve:'
    plot_learning_curve(lr,X_s,y,train_sizes=np.linspace(0.1,1,10),scoring='roc_auc')

    #重采样
    lr_r = LogisticRegression(penalty='l2',C=0.002,class_weight='balanced')
    lr_r.fit(X_s_resampled,y_s_resampled)

    predicted = lr_r.predict(test_X_s)
    predicted_proba = lr_r.predict_proba(test_X_s)

    print 'lr:'

    print "confusion_matrix:"
    c_matrix = metrics.confusion_matrix(test_y, predicted)
    plot_confusion_matrix(c_matrix)

    print "classification_report:"
    print metrics.classification_report(test_y, predicted)

    print "AUC:"
    plot_roc_curve(test_y,predicted_proba[:,1])

    print "KS:"
    plot_ks_curve(test_y,predicted_proba[:,1])

    print "PSI:"
    test1 = lr_r.predict(X_s)
    test2 = predicted
    bins = [0,0.5,1]
    print psi(test1,test2,bins)

    print 'learning curve:'
    plot_learning_curve(lr_r,X_s,y,train_sizes=np.linspace(0.1,1,10),scoring='roc_auc')

    #分箱
    lr_me = LogisticRegression(penalty='l2',C=0.0005,class_weight='balanced')
    lr_me.fit(X_me_s,y)

    predicted = lr_me.predict(test_X_me_s)
    predicted_proba = lr_me.predict_proba(test_X_me_s)

    print 'lr:'

    print "confusion_matrix:"
    c_matrix = metrics.confusion_matrix(test_y, predicted)
    plot_confusion_matrix(c_matrix)

    print "classification_report:"
    print metrics.classification_report(test_y, predicted)

    print "AUC:"
    plot_roc_curve(test_y,predicted_proba[:,1])

    print "KS:"
    plot_ks_curve(test_y,predicted_proba[:,1])

    print "PSI:"
    test1 = lr_me.predict(X_me_s)
    test2 = predicted
    bins = [0,0.5,1]
    print psi(test1,test2,bins)

    print 'learning curve:'
    plot_learning_curve(lr_me,X_me_s,y,train_sizes=np.linspace(0.1,1,10),scoring='roc_auc')






"""
import sys,os,time,pickle
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn import preprocessing
from sklearn.preprocessing import Imputer
from sklearn import cross_validation
from sklearn import feature_selection
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.lda import LDA


#Linear Regression
def linear_regression(train_X,train_y_r):
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(train_X,train_y_r)
    return model


# Logistic Regression Classifier
def logistic_regression_classifier(train_x, train_y):
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(penalty='l2')
    model.fit(train_x, train_y)
    return model

# Multinomial Naive Bayes Classifier
def naive_bayes_classifier(train_x, train_y):
    from sklearn.naive_bayes import MultinomialNB
    model = MultinomialNB(alpha=0.01)
    model.fit(train_x, train_y)
    return model

# KNN Classifier
def knn_classifier(train_x, train_y):
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier()
    model.fit(train_x, train_y)
    return model

# Random Forest Classifier
def random_forest_classifier(train_x, train_y):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=8)
    model.fit(train_x, train_y)
    return model

# Decision Tree Classifier
def decision_tree_classifier(train_x, train_y):
    from sklearn import tree
    model = tree.DecisionTreeClassifier()
    model.fit(train_x, train_y)
    return model



# GBDT(Gradient Boosting Decision Tree) Classifier
def gradient_boosting_classifier(train_x, train_y):
    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier(n_estimators=200)
    model.fit(train_x, train_y)
    return model

# SVM Classifier
def svm_classifier(train_x, train_y):
    from sklearn.svm import SVC
    model = SVC(kernel='rbf', probability=True)
    model.fit(train_x, train_y)
    return model

#Adaboost
def adaboost_classifier(train_x,train_y):
    from sklearn.ensemble import AdaBoostClassifier
    model = AdaBoostClassifier(n_estimators=200)
    model.fit(train_x,train_y)
    return model

def read_data(data_file):
    data = pd.read_csv(data_file)

    #划分训练集和测试集（分类）
    train_X, test_X, train_y, test_y = cross_validation.train_test_split(data[data.columns[2:]], data['SeriousDlqin2yrs']
    , test_size = 0.2, random_state = 0) #返回数据类型为dataframe


    # 缺失值处理（mean,median,most_frequent）
    imp = Imputer(missing_values='NaN', strategy='median', axis=0)
    imp.fit(train_X)
    train_X = imp.transform(train_X) # train_X数据类型变为ndarray
    test_X = imp.transform(test_X)
    ##train_X.dropna(how = 'any',inplace = True)

    #异常值处理
    olp = preprocessing.RobustScaler().fit(train_X)
    train_X = olp.transform(train_X) # transform函数本身对train_X没影响，必须赋值
    test_X = olp.transform(test_X)

   
    #标准化数据
    #scaler = preprocessing.StandardScaler().fit(train_X)
    scaler = preprocessing.MinMaxScaler().fit(train_X) # 归一化
    #scaler = preprocessing.Normalizer().fit(train_X) #正则化
    train_X = scaler.transform(train_X) # transform函数本身对train_X没影响，必须赋值
    test_X = scaler.transform(test_X)



    #特征选择
    #train_X_new = feature_selection.VarianceThreshold(threshold=3).fit_transform(train_X)#方差选择法
    #train_X_new = feature_selection.SelectKBest(feature_selection.f_regression, k = 10).fit_transform(train_X,train_y_r)#相关系数法
    fs = feature_selection.SelectPercentile(feature_selection.chi2, percentile = 80).fit(train_X,train_y)#train_X中数据需非负
    train_X = fs.transform(train_X)
    test_X = fs.transform(test_X)


    #降维

    #PCA
    dec = PCA(n_components=4).fit(train_X)
    train_X = dec.transform(train_X)
    test_X = dec.transform(test_X)

    #LDA 分类
    dec = LDA(n_components=1).fit(train_X,train_y_c)
    train_X = dec.transform(train_X)
    test_X = dec.transform(test_X)
    

    return train_X, train_y, test_X, test_y


if __name__ == '__main__':
    pwd = os.getcwd()
    data_file = pwd+"/credit_card_data.csv"
    #thresh = 0.5
    model_save_file = pwd+'/mode_save.txt'
    model_save = {}

    test_classifiers = [ 'NB','KNN', 'LR', 'RF', 'DT', 'SVM', 'Ada', 'GBDT']
    classifiers = {'NB': naive_bayes_classifier,
                   'KNN': knn_classifier,
                   'LR': logistic_regression_classifier,
                   'RF': random_forest_classifier,
                   'DT': decision_tree_classifier,
                   'SVM': svm_classifier,
                   'Ada': adaboost_classifier,
                   'GBDT': gradient_boosting_classifier
                   }

    print('reading training and testing data...')
    train_X, train_y, test_X, test_y = read_data(data_file)

    for classifier in test_classifiers:
        if classifier == 'SVM' or classifier == 'NB' or classifier == 'LR':
            continue
        print('******************* %s ********************' % classifier)
        start_time = time.time()
        model = classifiers[classifier](train_X, train_y)
        print('training took %fs!' % (time.time() - start_time))
        predicted = model.predict(test_X)
        if model_save_file != None:
            model_save[classifier] = model
        #print(metrics.classification_report(test_y_c, predicted))
        #print 'confusion_matrix:'
        #print (metrics.confusion_matrix(test_y_c, predicted))
        print 'AUC:', metrics.roc_auc_score(test_y, predicted)
        
        precision = metrics.precision_score(test_y_c, predict)
        recall = metrics.recall_score(test_y_c, predict)
        print('precision: %.2f%%, recall: %.2f%%' % (100 * precision, 100 * recall))
        accuracy = metrics.accuracy_score(test_y_c, predict)
        print('accuracy: %.2f%%' % (100 * accuracy))
       


    if model_save_file != None:
        pickle.dump(model_save, open(model_save_file, 'wb'))

"""
