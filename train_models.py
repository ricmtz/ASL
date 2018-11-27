from sklearn.externals import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from loader import load_data, reduce_dimen

TRAIN_DATA_PATH = 'sign-language-mnist/sign_mnist_train.csv'
TEST_DATA_PATH = 'sign-language-mnist/sign_mnist_test.csv'


def train_models(models, img, lbl):
    for name in models:
        print('Training {}'.format(name))
        models[name].fit(img, lbl)
        joblib.dump(models[name], '{}.joblib'.format(name))
        print('Training finish')


if __name__ == "__main__":
    classifiers = {
        'Knn': KNeighborsClassifier(n_neighbors=5),
        'SVC': SVC(kernel='poly'),
        'MLP': MLPClassifier(hidden_layer_sizes=(
            300, 150, 50), activation='tanh', solver='sgd', max_iter=500),
    }
    img_train, lbl_train = load_data(TRAIN_DATA_PATH)
    img_test, lbl_test = load_data(TEST_DATA_PATH)
    img_train = reduce_dimen(img_train)
    img_test = reduce_dimen(img_test)
    train_models(classifiers, img_train, lbl_train)
