from sklearn.externals import joblib
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import loader


TEST_DATA_PATH = 'sign-language-mnist/sign_mnist_test.csv'


def load_models():
    models = {
        'SVC': joblib.load('SVC.joblib'),
        'MLP': joblib.load('MLP.joblib'),
        'Knn': joblib.load('Knn.joblib'),
    }
    return models


def test_models(models, img, lbl):
    for name, model in models.items():
        print('Testing {}:'.format(name))
        lbl_pred = model.predict(img)
        score = model.score(img, lbl)
        res_f1_l = f1_score(lbl, lbl_pred, average='macro')
        res_f1_g = f1_score(lbl, lbl_pred, average='micro')
        con_m = confusion_matrix(lbl, lbl_pred)
        print('score: {}'.format(score))
        print('f1_score macro: {}'.format(res_f1_l))
        print('f1_score micro: {}'.format(res_f1_g))
        plt.matshow(con_m)
        plt.colorbar()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.title('Confusion matriz: {}'.format(name))
    plt.show()


if __name__ == "__main__":
    models = load_models()
    img, lbl = loader.load_data(TEST_DATA_PATH)
    img = loader.reduce_dimen(img)
    test_models(models, img, lbl)
