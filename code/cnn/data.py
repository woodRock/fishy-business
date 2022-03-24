from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
import scipy.io

def encode_labels(y):
    """
    Convert text labels to numbers.
    """
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)
    return y, le

def load(filename, folder=''):
    """
    Load the data from the mat file.
    """
    path = folder + filename
    mat = scipy.io.loadmat(path)
    return mat

def prepare(mat):
    """
    Load the data from matlab format into memory. 
    """
    X = mat['X']   
    X = X.astype(float)
    y = mat['Y']    
    y = y[:, 0]
    return X,y

def normalize(X_train, X_test):
    """
    Normalize the input features within range [0,1].
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test