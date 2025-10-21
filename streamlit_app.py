

import streamlit as st
from keras import Sequential
from keras.layers import Embedding, Dense, GlobalAveragePooling1D


st.title("Modèle Word2Vec test streamlit")

embedding_dim = 300
model = Sequential()
model.add(Embedding(vocab_size, embedding_dim))
model.add(GlobalAveragePooling1D())
model.add(Dense(vocab_size, activation='softmax'))

model.load_weights("word2vec.h5")

vectors = model.layers[0].trainable_weights[0].numpy()
import numpy as np
from sklearn.preprocessing import Normalizer

def dot_product(vec1, vec2):
    return np.sum((vec1*vec2))

def cosine_similarity(vec1, vec2):
    return dot_product(vec1, vec2)/np.sqrt(dot_product(vec1, vec1)*dot_product(vec2, vec2))

def find_closest(word_index, vectors, number_closest):
    list1=[]
    query_vector = vectors[word_index]
    for index, vector in enumerate(vectors):
        if not np.array_equal(vector, query_vector):
            dist = cosine_similarity(vector, query_vector)
            list1.append([dist,index])
    return np.asarray(sorted(list1,reverse=True)[:number_closest])

def compare(index_word1, index_word2, index_word3, vectors, number_closest):
    list1=[]
    query_vector = vectors[index_word1] - vectors[index_word2] + vectors[index_word3]
    normalizer = Normalizer()
    query_vector =  normalizer.fit_transform([query_vector], 'l2')
    query_vector= query_vector[0]
    for index, vector in enumerate(vectors):
        if not np.array_equal(vector, query_vector):
            dist = cosine_similarity(vector, query_vector)
            list1.append([dist,index])
    return np.asarray(sorted(list1,reverse=True)[:number_closest])

def print_closest(word, number=10):
    index_closest_words = find_closest(word2idx[word], vectors, number)
    for index_word in index_closest_words :
        print(idx2word[index_word[1]]," -- ",index_word[0])
        

#Exemple d'utilisation de la fonction print_closest
print_closest('zombie')

################################
        
# df=pd.read_csv("train.csv")

# st.title("Projet de classification binaire Titanic")
# st.sidebar.title("Sommaire")
# pages=["Exploration", "DataVizualization", "Modélisation"]
# page=st.sidebar.radio("Aller vers", pages)

# if page == pages[0] : 

    # st.write("### Introduction")
    # st.dataframe(df.head(10))
    # st.write(df.shape)
    # st.dataframe(df.describe())

    # if st.checkbox("Afficher les NA") :
        # st.dataframe(df.isna().sum())

# if page == pages[1] : 
    # st.write("### DataVizualization")

    # fig = plt.figure()
    # sns.countplot(x = 'Survived', data = df)
    # st.pyplot(fig)

    # fig = plt.figure()
    # sns.countplot(x = 'Sex', data = df)

    # plt.title("Répartition du genre des passagers")
    # st.pyplot(fig)

    # fig = plt.figure()
    # sns.countplot(x = 'Pclass', data = df)
    # plt.title("Répartition des classes des passagers")
    # st.pyplot(fig)

    # fig = sns.displot(x = 'Age', data = df)
    # plt.title("Distribution de l'âge des passagers")
    # st.pyplot(fig)

    # fig = plt.figure()
    # sns.countplot(x = 'Survived', hue='Sex', data = df)
    # st.pyplot(fig)

    # fig = sns.catplot(x='Pclass', y='Survived', data=df, kind='point')
    # st.pyplot(fig)

    # fig = sns.lmplot(x='Age', y='Survived', hue="Pclass", data=df)
    # st.pyplot(fig)

    # fig, ax = plt.subplots()
    # sns.heatmap(df.select_dtypes(include=[np.number]).corr(), ax=ax)
    # st.write(fig)

# if page == pages[2] : 
    # st.write("### Modélisation")

    # df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

    # y = df['Survived']
    # X_cat = df[['Pclass', 'Sex',  'Embarked']]
    # X_num = df[['Age', 'Fare', 'SibSp', 'Parch']]

    # for col in X_cat.columns:
        # X_cat[col] = X_cat[col].fillna(X_cat[col].mode()[0])

    # for col in X_num.columns:
        # X_num[col] = X_num[col].fillna(X_num[col].median())
        # X_cat_scaled = pd.get_dummies(X_cat, columns=X_cat.columns)
        # X = pd.concat([X_cat_scaled, X_num], axis = 1)

    # from sklearn.model_selection import train_test_split
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    # from sklearn.preprocessing import StandardScaler
    # scaler = StandardScaler()
    # X_train[X_num.columns] = scaler.fit_transform(X_train[X_num.columns])
    # X_test[X_num.columns] = scaler.transform(X_test[X_num.columns])

    # from sklearn.ensemble import RandomForestClassifier
    # from sklearn.svm import SVC
    # from sklearn.linear_model import LogisticRegression
    # from sklearn.metrics import confusion_matrix

    # def prediction(classifier):
        # if classifier == 'Random Forest':
            # clf = RandomForestClassifier()
        # elif classifier == 'SVC':
            # clf = SVC()
        # elif classifier == 'Logistic Regression':
            # clf = LogisticRegression()
        # clf.fit(X_train, y_train)
        # return clf

    # def scores(clf, choice):
        # if choice == 'Accuracy':
            # return clf.score(X_test, y_test)
        # elif choice == 'Confusion matrix':
            # return confusion_matrix(y_test, clf.predict(X_test))

    # choix = ['Random Forest', 'SVC', 'Logistic Regression']
    # option = st.selectbox('Choix du modèle', choix)
    # st.write('Le modèle choisi est :', option)

    # clf = prediction(option)
    # display = st.radio('Que souhaitez-vous montrer ?', ('Accuracy', 'Confusion matrix'))
    # if display == 'Accuracy':
        # st.write(scores(clf, display))
    # elif display == 'Confusion matrix':
        # st.dataframe(scores(clf, display))