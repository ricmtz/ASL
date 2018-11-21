from sklearn.externals import joblib
from sklearn.neighbors import KNeighborsClassifier
from loader import load_data

TRAIN_DATA_PATH = 'sign-language-mnist/sign_mnist_train.csv'
TEST_DATA_PATH = 'sign-language-mnist/sign_mnist_test.csv'


def train_knn(img, lbl):
    model = KNeighborsClassifier(n_neighbors=15)
    model.fit(img, lbl)
    joblib.dump(model, 'Knn.joblib')
    return model


if __name__ == "__main__":
    img, lbl = load_data(TRAIN_DATA_PATH)
    img_t, lbl_t = load_data(TEST_DATA_PATH)
    model = train_knn(img, lbl)
    print(model.score(img_t, lbl_t))
