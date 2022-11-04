from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
import pandas as pd
import numpy as np
import pickle
from flask import Flask,request,jsonify,render_template


df = pd.read_csv('pizza_data.csv')
df.columns = ['pizza_company', 'pizza_name',
              'pizza_type', 'pizza_size', 'pizza_price']


LE1 = LabelEncoder()
df.pizza_company = LE1.fit_transform(df.pizza_company)
LE2 = LabelEncoder()
df.pizza_name = LE2.fit_transform(df.pizza_name)
LE3 = LabelEncoder()
df.pizza_type = LE3.fit_transform(df.pizza_type)
LE4 = LabelEncoder()
df.pizza_size = LE4.fit_transform(df.pizza_size)
LE5 = LabelEncoder()
df.pizza_price = LE5.fit_transform(df.pizza_price)

# X = np.array(df.drop(['pizza_price'], axis=1))
# y = np.array(df.pizza_price)
# DT_classifier = DecisionTreeClassifier()
# DT_classifier.fit(X, y)


filename = 'finalized_model.sav'
# pickle.dump(DT_classifier, open(filename, 'wb'))


def tell_me_price(pizza_company, pizza_name, pizza_type, pizza_size):
    v1 = LE1.classes_
    v2 = LE2.classes_
    v3 = LE3.classes_
    v4 = LE4.classes_
    ind1 = np.where(v1 == pizza_company)[0]
    ind2 = np.where(v2 == pizza_name)[0]
    ind3 = np.where(v3 == pizza_type)[0]
    ind4 = np.where(v4 == pizza_size)[0]
    v = (ind1, ind2, ind3, ind4)
    if(v[0].size==0 or v[1].size==0 or v[2].size==0 or v[3].size==0):
        return '+-+-'
    loaded_model = pickle.load(open(filename, 'rb'))
    price_index = loaded_model.predict(
        [np.array([v[0][0], v[1][0], v[2][0], v[3][0]])])
    prices = LE5.classes_
    return (prices[price_index][0])


# print(tell_me_price("IMO's Pizza", "BBQ Chicken Pizza",
#       "Specialty Pizzas", 'X Large (16")'))



app = Flask(__name__)

@app.route('/',methods=['GET','POST'])
def Home():
    prediction_model='+-+-'
    if request.method=='POST':
        r1=request.form['pizza_company']
        r2=request.form['pizza_name']
        r3=request.form['pizza_type']
        r4=request.form['pizza_size']
        prediction_model=tell_me_price(r1,r2,r3,r4)

    return render_template('index.html',prediction_text=prediction_model)



@app.route('/products')
def products():
    return render_template('products.html')





if __name__=="__main__":
    app.run(debug=True)
