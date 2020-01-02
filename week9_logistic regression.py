''' version 1,date: 31/12 20:30'''
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


class logisticRegrssion():

    def __init__(self,lr=0.1,n_iteration=1000,treshold=0.005,method='GD',track_loss=True):
        super().__init__()
        self.lr=lr
        self.n_iteration=n_iteration
        self.treshold=treshold
        self.method=method
        self.track_loss=track_loss
        self.weights = 0
        self.theta = 0
        self.grad_loss = []

    def fit(self, X, y):
        self.X=X
        self.y=y.reshape(-1, 1)
        self.weights=np.zeros((self.X.shape[1]))
        self.gradientdescent(self.X, self.y)

    def predict(self, X):
        y_pred=1/(1+np.exp(-X@self.weights))
        y_pred[y_pred > 0.5] = 1
        y_pred[y_pred < 0.5] = 0
        return y_pred


    def predict_prob(self, X):
        y_pred=1/(1+np.exp(-X@self.weights))
        return y_pred


    def gradientdescent(self, X, y):
        self.weights_new = np.random.uniform(low=0, high=1, size=self.X.shape[1]).reshape(-1,1)
        self.weights_old = np.random.uniform(low=0, high=1, size=self.X.shape[1]).reshape(-1,1)

        y=y.reshape(-1,1)
        N=y.shape[0]
        for i in range(self.n_iteration):
            self.updateWeights(X, y,self.weights_new)
            Loss=-((1/N)*(y.T@np.log(self.theta)+(1-y).T@np.log(1-self.theta)))
            if self.track_loss:
                self.grad_loss.append(Loss[0][0])
            if (i % (100)) == 0:
                print('Loss is:', Loss[0][0])
        self.weights = self.weights_new
        return

    def updateWeights(self, X, y,w):
        N=X.shape[0]
        self.weight_old=w.reshape(-1,1)
        w_0=self.weight_old
        self.theta = 1 / (1 + np.exp(-(X@w_0)))
        if self.method=='GD':
            self.weights_new=w_0-self.lr*(1/N)*X.T @(self.theta-y)
        elif self.method=='SGD':
            i=np.random.randint(0,X.shape[0])
            self.weights_new=w_0-(self.lr*X[i]*(self.theta[i]-y[i])).reshape(-1,1)
        # elif self.method=='NM':
        #     norm=self.lr * (1 / N)
        #     H=-((self.theta).T@(1-self.theta)*((X.T@X)))#-(((self.theta)*(1-self.theta)).T*((X@X.T)[0])).T
        #     H_inv=np.linalg.inv(H)
        #     self.weights_new =(w_0 + norm*H_inv @(X.T @(y-self.theta)))
        return

    def score(self,y_actual,y_pred,method='MSE'):
        global cost
        if method=='MSE':
            N=y_actual.shape[0]
            cost =(1/(N))*np.sum(np.square(y_actual-y_pred))
            #MSEKLEARN=sklearn.metrics.mean_squared_error(y_actual,y_pred)
        elif method=='Mape':
            N = y_actual.shape[0]
            cost = np.sum(np.absolute(y_actual - y_pred))/N
        elif method == 'TPR':
            cost = 100*(np.sum(y_pred[y_pred==1])/np.sum(y_actual[y_actual==1]))

        return cost

if __name__ == "__main__":
    '''loading data'''

    # #part 2
    #
    # from sklearn.datasets import load_iris
    # iris = load_iris()
    # X = iris.data[:, :2]
    # y = (iris.target != 0) * 1
    # y = y.reshape(-1, 1)
    # '''splitting data train and test'''
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
    #                                                     random_state=10)
    # '''normalizing and scaling '''
    # ssx = StandardScaler().fit(X_train)
    # X_train_std = ssx.transform(X_train)
    # X_test_std = ssx.transform(X_test)
    # #GD
    # LogRegGD = logisticRegrssion(method='GD',n_iteration=1000)
    # LogRegGD.fit(X_train_std, y_train)
    # # train
    # y_pred_train = LogRegGD.predict(X_train_std)
    # Base_MSE_train_GD = LogRegGD.score(y_train, y_pred_train,method="TPR")
    # # test
    # y_pred_test= LogRegGD.predict(X_test_std)
    # Base_MSE_test_GD = LogRegGD.score(y_test, y_pred_test,method="TPR")
    # plt.plot(list(range(LogRegGD.n_iteration)),LogRegGD.grad_loss)
    # plt.xlabel('epoch')
    # plt.ylabel('loss')
    # plt.title('GD loss ')
    # plt.show()
    # #SGD
    # LogRegSGD = logisticRegrssion(method='SGD',n_iteration=1000 )
    # LogRegSGD.fit(X_train_std, y_train)
    # # train
    # y_pred_train = LogRegSGD.predict(X_train_std)
    # Base_MSE_train_SGD = LogRegSGD.score(y_train, y_pred_train, method="TPR")
    # # test
    # y_pred_test = LogRegSGD.predict(X_test_std)
    # Base_MSE_test_SGD = LogRegSGD.score(y_test, y_pred_test, method="TPR")
    # plt.plot( range(LogRegSGD.n_iteration),LogRegSGD.grad_loss)
    # plt.xlabel('epoch')
    # plt.ylabel('loss')
    # plt.title('SGD loss ')
    # plt.show()
    #
    #
    # #part 3
    #
    # import matplotlib.pyplot as plt
    # def plot_samples(X, y, gd_model=None, sgd_model=None):
    #     plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='b', label='0')
    #     plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='r', label='1')
    #
    #     x1_min, x1_max = X[:, 0].min(), X[:, 0].max(),
    #     x2_min, x2_max = X[:, 1].min(), X[:, 1].max(),
    #     xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
    #     grid = np.c_[xx1.ravel(), xx2.ravel()]
    #     if gd_model is not None:
    #         probs = gd_model.predict_prob(grid).reshape(xx1.shape)
    #         plt.contour(xx1, xx2, probs, [0.5], linewidths=1, colors='black', label='GD');
    #     if sgd_model is not None:
    #         probs = sgd_model.predict_prob(grid).reshape(xx1.shape)
    #         plt.contour(xx1, xx2, probs, [0.5], linewidths=1, colors='green', label='SGD');
    #     plt.legend()
    #     plt.show()
    #     print('poop')
    #
    #
    # LogRegGD1 = logisticRegrssion(method='GD', n_iteration=1000)
    # LogRegGD1.fit(X, y)
    # LogRegSGD1 = logisticRegrssion(method='SGD', n_iteration=1000)
    # LogRegSGD1.fit(X, y)
    # plot_samples(X, y, gd_model=LogRegGD1, sgd_model=LogRegSGD1)
    #


    #part 4-mnist

    import sklearn
    from sklearn.datasets import fetch_openml
    import pandas  as pd
    from sklearn.datasets import fetch_mldata
    import tempfile
    test_data_home = tempfile.mkdtemp()
    #mnist = fetch_mldata('MNIST original', transpose_data=True, data_home=test_data_home)
    mnist = fetch_mldata('MNIST original', transpose_data=True, data_home='./')
    x = mnist.data
    y = mnist.target
    #x = x / 255.0 * 2 - 1
    data = np.concatenate((x, y.reshape(-1,1)), axis=1)
    data=pd.DataFrame(data)
    data.to_csv('./mnist.csv')
    data=pd.read_csv('./mnist.csv')
    X,y=data.iloc[:,-1].values,data.iloc[:,-1].values
    X,y=X.reshape(-1,1),y.reshape(-1,1)
    x_train, x_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=(1/7),
                                                        random_state=42)
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix, classification_report

    Log=LogisticRegression(solver='lbfgs',multi_class='multinomial')
    Log.fit(x_train.reshape(-1,1),y_train.reshape(-1,1))
    y_test_pred=Log.predict(x_test.reshape(-1,1))
    cm=confusion_matrix(y_test, y_test_pred)
    print(classification_report(y_test, y_test_pred))
    print('poop')