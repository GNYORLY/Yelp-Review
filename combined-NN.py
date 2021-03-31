from sklearn.neural_network import MLPClassifier
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA

MAX_ITER = 1
MEAN = 3.4
MEAN2 = 4.0


train  = pd.read_csv('data/train.csv')
X_train = train.drop('review_stars',axis=1)
y_train = train[['review_stars']].copy()


vali = pd.read_csv('data/validate.csv')
X_vali = vali.drop('review_stars',axis=1)
y_vali = vali[['review_stars']].copy()
ohe = OneHotEncoder(sparse=False)
y_train_transformed = ohe.fit_transform(y_train).transpose()
y_vali_transformed = ohe.fit_transform(y_vali).transpose()
y_train_1 = y_train_transformed[0]
y_train_2 = y_train_transformed[1]
y_train_3 = y_train_transformed[2]
y_train_4 = y_train_transformed[3]
y_train_5 = y_train_transformed[4]
y_vali_1 = y_vali_transformed[0]
y_vali_2 = y_vali_transformed[1]
y_vali_3 = y_vali_transformed[2]
y_vali_4 = y_vali_transformed[3]
y_vali_5 = y_vali_transformed[4]

#pca = PCA(n_components=3)
#X_train = pca.fit_transform(X_train)
#X_vali = pca.fit_transform(X_vali)


y_vali_transformed = ohe
def create_result_df(prediction):
    return pd.DataFrame(data={
        'index': np.arange(len(prediction)),
        'stars': prediction
    })

scaler = StandardScaler()

# Fit only to the training data
scaler.fit(X_train)
StandardScaler(copy=True, with_mean=True, with_std=True)

# Now apply the transformations to the data:
X_train = scaler.transform(X_train)
X_vali = scaler.transform(X_vali)

mlp1 = MLPClassifier(hidden_layer_sizes=(30,6), max_iter=MAX_ITER, verbose=False, activation='identity')
mlp1.fit(X_train,y_train_1)
mlp2 = MLPClassifier(hidden_layer_sizes=(30,6), max_iter=MAX_ITER, verbose=False, activation='identity')
mlp2.fit(X_train,y_train_2)
mlp3 = MLPClassifier(hidden_layer_sizes=(30,6), max_iter=MAX_ITER, verbose=False, activation='identity')
mlp3.fit(X_train,y_train_3)
mlp4 = MLPClassifier(hidden_layer_sizes=(30,6), max_iter=MAX_ITER, verbose=False, activation='identity')
mlp4.fit(X_train,y_train_4)
mlp5 = MLPClassifier(hidden_layer_sizes=(30,6), max_iter=MAX_ITER, verbose=False, activation='identity')
mlp5.fit(X_train,y_train_5)
mlp = MLPClassifier(hidden_layer_sizes=(200,50,30,20,10), max_iter=MAX_ITER, verbose=False)
mlp.fit(X_train,y_train)
pred1 = mlp1.predict(X_vali)
pred2 = mlp2.predict(X_vali)
pred3 = mlp3.predict(X_vali)
pred4 = mlp4.predict(X_vali)
pred5 = mlp5.predict(X_vali)
pred = mlp.predict(X_vali)


#print(classification_report(y_vali_1,predictions))

print("accuracy1:", mlp1.score(X_vali, y_vali_1))
print("accuracy2:", mlp2.score(X_vali, y_vali_2))
print("accuracy3:", mlp3.score(X_vali, y_vali_3))
print("accuracy4:", mlp4.score(X_vali, y_vali_4))
print("accuracy5:", mlp5.score(X_vali, y_vali_5))
print("")

root_mean_absolute_error = np.sqrt(mean_absolute_error(y_vali, pred))
print("accurancy:", mlp.score(X_vali, y_vali))
print("root_mean_absolute_error:", root_mean_absolute_error)


pred_combined = pred.copy()

for i in range(len(pred)):
    bin_sum = pred1[i]+pred2[i]+pred3[i]+pred4[i]+pred5[i]
    if ( (pred1[i] == 1 and pred_combined[i] == 1)
         or (pred2[i] == 1 and pred_combined[i] == 2)
         or (pred3[i] == 1 and pred_combined[i] == 3)
         or (pred4[i] == 1 and pred_combined[i] == 4)
         or (pred5[i] == 1 and pred_combined[i] == 5) ):
        if bin_sum == 1:
            #most confident
            pred_combined[i] = (.3*MEAN+pred_combined[i])/1.3
        else:
            #quite confident
            pred_combined[i] = (pred1[i]+
                                2*pred2[i]+
                                3*pred3[i]+
                                4*pred4[i]+
                                5*pred5[i]+
                                1*pred_combined[i]+MEAN)  /  (2+bin_sum)
            
        
    elif bin_sum == 0:
        #least confidence
        pred_combined[i] = (1.8*MEAN+pred_combined[i])/2.8

    else:
        #some confidence
        #average out the models
        pred_combined[i] = (pred1[i]+
                            2*pred2[i]+
                            3*pred3[i]+
                            4*pred4[i]+
                            5*pred5[i]+
                            pred_combined[i]+MEAN)  /  (2+bin_sum)




        

root_mean_absolute_error = np.sqrt(mean_absolute_error(y_vali, pred_combined))
squared_error = np.sqrt(mean_squared_error(y_vali, pred_combined))
print("absolute_error:", root_mean_absolute_error)
print("squared_error:", squared_error)



submission_df = create_result_df(pred_combined)
submission_df.to_csv("./data/result.csv", encoding='utf-8', index=False)
