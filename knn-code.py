# =========================TUPRO3================================
# Project           : Tugas Pemrograman 3 Sistem Cerdas         =
# Kelas             : IT-43-03                                  =
# Anggota Kelompok  : Bagas Alfito Prismawan    (1303193027)    =
#                     Kevin Antonio Fajrin      (1303193123)    =
#                     Sultan Kautsar            (1303194010)    =
# ===============================================================

# Import library
import pandas as pd
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# Import data
data_train = pd.read_excel('DataSetTB3_SHARE.xlsx', sheet_name='Data')
data_test = pd.read_excel('DataSetTB3_SHARE.xlsx', sheet_name='Submit')

data_train.head(15)

# Describe().T
var: object = data_train.describe().T

# Pre-processing dan split data
miss_values = data_train.columns[data_train.isnull().any()]
print(f"Missing values:\n{data_train[miss_values].isnull().sum()}")

null_values = data_train.columns[data_train.isna().any()]
print(f"Null Values:\n{data_train[null_values].isna().sum()}")

# Diperlukan normalisasi karena terdapat 'null'
# Normalisasi
columns = ['label', ['pixel' + str(i) for i in range(785)]]
scale_X = StandardScaler()
X = pd.DataFrame(scale_X.fit_transform(data_train.drop(["label"], axis=1), ), columns=columns[1])

X.head(20)

y = pd.DataFrame(data_train, columns=['label'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

var2 = X_train.shape, X_test.shape, y_train.shape, y_test.shape

# Modeling dan Train
# euclidean
knn = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, metric='euclidean')
knn.fit(X_train, y_train)

# Manhattan
knn1: KNeighborsClassifier = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30,
                                                  metric='manhattan')
knn1.fit(X_train, y_train)

# Chebyshev
knn2 = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, metric='chebyshev')
knn2.fit(X_train, y_train)

# Minkowski
knn3 = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, metric='minkowski')
knn3.fit(X_train, y_train)

# Evaluation
# Euclidean
y_pred_knn = knn.predict(X_test)

knn = metrics.confusion_matrix(y_test, y_pred_knn)
print(knn)
print(classification_report(y_test, y_pred_knn))

# Manhattan
y_pred_knn1 = knn1.predict(X_test)

knn11 = metrics.confusion_matrix(y_test, y_pred_knn1)
print(knn11)
print(classification_report(y_test, y_pred_knn1))

# Chebyshev
y_pred_knn2 = knn2.predict(X_test)

knn2 = metrics.confusion_matrix(y_test, y_pred_knn2)
print(knn2)
print(classification_report(y_test, y_pred_knn2))

# Minkowski
y_pred_knn3 = knn3.predict(X_test)

knn3 = metrics.confusion_matrix(y_test, y_pred_knn3)
print(knn3)
print(classification_report(y_test, y_pred_knn3))

accuracy1 = metrics.accuracy_score(y_test, y_pred_knn)
accuracy2 = metrics.accuracy_score(y_test, y_pred_knn1)
accuracy3 = metrics.accuracy_score(y_test, y_pred_knn2)
accuracy4 = metrics.accuracy_score(y_test, y_pred_knn3)

print("Euclidean:", accuracy1)
print("Manhattan:", accuracy2)
print("Chebyshev:", accuracy3)
print("Minkowski:", accuracy4)

# Melakukan pemilihan manhattan (tertinggi)
saved_model = pickle.dumps(knn1)

knn1_from_pickle = pickle.loads(saved_model)

knn1_array = knn1_from_pickle.predict(X_test)
print(len(knn1_array))

out_data = {'idData': [i for i in range(len(knn1_array) - 1)],
            'Klasifikasi': [knn1_array[i] for i in range(len(knn1_array) - 1)],
            'Akurasi': accuracy1 * 100
            }
out = pd.DataFrame(out_data, columns=['idData', 'Klasifikasi', 'Akurasi'])
out.to_excel("OutputLatih.xlsx")
print(out)

# Submit
# Melakukan pengujian classifier 2-5
# Euclidean
knn_e2 = KNeighborsClassifier(n_neighbors=2, weights='uniform', algorithm='auto', leaf_size=30, metric="euclidean")
knn_e3 = KNeighborsClassifier(n_neighbors=3, weights='uniform', algorithm='auto', leaf_size=30, metric="euclidean")
knn_e4 = KNeighborsClassifier(n_neighbors=4, weights='uniform', algorithm='auto', leaf_size=30, metric="euclidean")
knn_e5 = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, metric="euclidean")

# Manhattan
knn_m2 = KNeighborsClassifier(n_neighbors=2, weights='uniform', algorithm='auto', leaf_size=30, metric="manhattan")
knn_m3 = KNeighborsClassifier(n_neighbors=3, weights='uniform', algorithm='auto', leaf_size=30, metric="manhattan")
knn_m4 = KNeighborsClassifier(n_neighbors=4, weights='uniform', algorithm='auto', leaf_size=30, metric="manhattan")
knn_m5 = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, metric="manhattan")

# Chebyshev
knn_c2 = KNeighborsClassifier(n_neighbors=2, weights='uniform', algorithm='auto', leaf_size=30, metric="chebyshev")
knn_c3 = KNeighborsClassifier(n_neighbors=3, weights='uniform', algorithm='auto', leaf_size=30, metric="chebyshev")
knn_c4 = KNeighborsClassifier(n_neighbors=4, weights='uniform', algorithm='auto', leaf_size=30, metric="chebyshev")
knn_c5 = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, metric="chebyshev")

# Minkowski
knn_w2 = KNeighborsClassifier(n_neighbors=2, weights='uniform', algorithm='auto', leaf_size=30, metric="minkowski")
knn_w3 = KNeighborsClassifier(n_neighbors=3, weights='uniform', algorithm='auto', leaf_size=30, metric="minkowski")
knn_w4 = KNeighborsClassifier(n_neighbors=4, weights='uniform', algorithm='auto', leaf_size=30, metric="minkowski")
knn_w5 = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, metric="minkowski")

# Euclidean
knn_e2.fit(X_train, y_train)
knn_e3.fit(X_train, y_train)
knn_e4.fit(X_train, y_train)
knn_e5.fit(X_train, y_train)

# Manhattan
knn_m2.fit(X_train, y_train)
knn_m3.fit(X_train, y_train)
knn_m4.fit(X_train, y_train)
knn_m5.fit(X_train, y_train)

# Chebyshev
knn_c2.fit(X_train, y_train)
knn_c3.fit(X_train, y_train)
knn_c4.fit(X_train, y_train)
knn_c5.fit(X_train, y_train)

# Minkowski
knn_w2.fit(X_train, y_train)
knn_w3.fit(X_train, y_train)
knn_w4.fit(X_train, y_train)
knn_w5.fit(X_train, y_train)

# Prediksi
# Euclidean
y_pred_knn_e2 = knn_e2.predict(X_test)
y_pred_knn_e3 = knn_e3.predict(X_test)
y_pred_knn_e4 = knn_e4.predict(X_test)
y_pred_knn_e5 = knn_e5.predict(X_test)

# Manhattan
y_pred_knn_m2 = knn_m2.predict(X_test)
y_pred_knn_m3 = knn_m3.predict(X_test)
y_pred_knn_m4 = knn_m4.predict(X_test)
y_pred_knn_m5 = knn_m5.predict(X_test)

# Chebyshev
y_pred_knn_c2 = knn_c2.predict(X_test)
y_pred_knn_c3 = knn_c3.predict(X_test)
y_pred_knn_c4 = knn_c4.predict(X_test)
y_pred_knn_c5 = knn_c5.predict(X_test)

# Minkowski
y_pred_knn_w2 = knn_w2.predict(X_test)
y_pred_knn_w3 = knn_w3.predict(X_test)
y_pred_knn_w4 = knn_w4.predict(X_test)
y_pred_knn_w5 = knn_w5.predict(X_test)

# Akurasi
# Euclidean
accuracy_knn_e2 = metrics.accuracy_score(y_test, y_pred_knn_e2)
accuracy_knn_e3 = metrics.accuracy_score(y_test, y_pred_knn_e3)
accuracy_knn_e4 = metrics.accuracy_score(y_test, y_pred_knn_e4)
accuracy_knn_e5 = metrics.accuracy_score(y_test, y_pred_knn_e5)

# Manhattan
accuracy_knn_m2 = metrics.accuracy_score(y_test, y_pred_knn_m2)
accuracy_knn_m3 = metrics.accuracy_score(y_test, y_pred_knn_m2)
accuracy_knn_m4 = metrics.accuracy_score(y_test, y_pred_knn_m2)
accuracy_knn_m5 = metrics.accuracy_score(y_test, y_pred_knn_m2)

# Chebyshev
accuracy_knn_c2 = metrics.accuracy_score(y_test, y_pred_knn_c2)
accuracy_knn_c3 = metrics.accuracy_score(y_test, y_pred_knn_c3)
accuracy_knn_c4 = metrics.accuracy_score(y_test, y_pred_knn_c4)
accuracy_knn_c5 = metrics.accuracy_score(y_test, y_pred_knn_c5)

# Minkowski
accuracy_knn_w2 = metrics.accuracy_score(y_test, y_pred_knn_w2)
accuracy_knn_w3 = metrics.accuracy_score(y_test, y_pred_knn_w3)
accuracy_knn_w4 = metrics.accuracy_score(y_test, y_pred_knn_w4)
accuracy_knn_w5 = metrics.accuracy_score(y_test, y_pred_knn_w5)

print("Euclidean:", accuracy_knn_e2)
print("Euclidean:", accuracy_knn_e3)
print("Euclidean:", accuracy_knn_e4)
print("Euclidean:", accuracy_knn_e5)

print("Manhattan:", accuracy_knn_m2)
print("Manhattan:", accuracy_knn_m3)
print("Manhattan:", accuracy_knn_m4)
print("Manhattan:", accuracy_knn_m5)

print("Chebyshev:", accuracy_knn_c2)
print("Chebyshev:", accuracy_knn_c3)
print("Chebyshev:", accuracy_knn_c4)
print("Chebyshev:", accuracy_knn_c5)

print("Minkowski:", accuracy_knn_w2)
print("Minkowski:", accuracy_knn_w3)
print("Minkowski:", accuracy_knn_w4)
print("Minkowski:", accuracy_knn_w5)

# Save model pada manhattan 5
saved = pickle.dumps(knn_m5)
knn_m5_from_pickle = pickle.loads(saved)
knn_m5_array = knn_m5_from_pickle.predict(data_test)
print(knn_m5_array)

# Print 10 kolom saja
maxten = 200
out_data2 = {'idData': [i for i in range(maxten)],
             'Klasifikasi': [knn_m5_array[i] for i in range(maxten)]
             }
out2 = pd.DataFrame(out_data2, columns=['idData', 'Klasifikasi'])
out2.to_excel("OuputSubmit.xlsx")
print(out2)
