import csv
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt


date_data = []
prices_data =[]

def get_data(filename):
    with open(filename,'r') as csvfile:
        csvFileReader = csv.reader(csvfile)
        next(csvFileReader)
        for row in csvFileReader:
            date_data.append(int(row[0].split('-')[0]))
            prices_data.append(float(row[1]))

    return

def predict_prices(data,prices,x):
    date_data = dates = np.reshape(date_data,(len(data),1))

    svr_lin = SVR(kernel= 'linear', C=1e3)
    svr_poly = SVR(kernel= 'poly', C=1e3, degree = 2)
    svr_rbf = SVR(kernel = 'rbf', C=1e3, gamma =0.1)

    svr_lin.fit(date_data,prices_data)
    svr_poly.fit(date_data,prices_data)
    svr_rbf.fit(date_data,prices_data)

    plt.scatter(date_data,prices_data, color='black', label ='Data')
    plt.plot(dates, svr_rbf.predict(date_data), color ='red', label = 'RBF Model')
    plt.plot(dates, svr_lin.predict(date_data), color='green', label= "Linear Model")
    plt.plot(dates, svr_poly.predict(date_data), color= "blue", label= "Polynomial Model")
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Support Vector Regression')
    plt.legend()
    plt.show()

    return svr_rbf.predict(x)[0], svr_lin.predict(x)[0], svr_poly.predict(x)[0]

get_data('aapl.csv')

predicted_price = predict_prices(date_data,prices_data,35)


print(predicted_price)