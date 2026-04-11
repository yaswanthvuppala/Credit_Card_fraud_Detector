# %%
import pandas as pd

# %%
df=pd.read_csv('creditcard.csv')
df.head()

# %%
X=df.drop(['Class'],axis=1)
Y=df['Class']

# %%
from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=30)

# %% [markdown]
# from imblearn.over_sampling import SMOTE
# 
# sm=SMOTE(k_neighbors=5,random_state=30)
# X_train_res,Y_train_res=sm.fit_resample(X_train,Y_train)

# %%
print(Y_train.value_counts())


# %% [markdown]
# from sklearn.preprocessing import StandardScaler
# 
# scaler=StandardScaler()
# 
# X_train=scaler.fit_transform(X_train)
# X_test=scaler.transform(X_test)

# %%
from xgboost import XGBClassifier 
 
ratio = len(Y_train[Y_train==0]) / len(Y_train[Y_train==1])

model = XGBClassifier(
    n_estimators=1200,
    max_depth=6,
    learning_rate=0.05,
    scale_pos_weight=ratio,
    random_state=42
)

model.fit(X_train,Y_train)

# %%

Y_pred=model.predict(X_test)

# %%
from sklearn.metrics import confusion_matrix,classification_report

print(classification_report(Y_test,Y_pred))
confusion_matrix(Y_test,Y_pred)

# %%


# %%
# from sklearn.metrics import precision_recall_curve

# precision, recall, thresholds = precision_recall_curve(Y_test, y_prob)

# %%



