from flask import Flask,render_template,url_for,request,redirect
from sklearn.preprocessing import LabelEncoder,StandardScaler
import pickle
import numpy as np

app=Flask(__name__)

@app.route('/',methods=["GET"])
def index():
    return render_template('index.html')


@app.route('/predict',methods=["POST","GET"])
def predict():
    if request.method=='POST':
        
        list=['gender', 'ever_married','residence','work_type','smoking_status','age', 'heart_disease', 'hypertension', 'bmi','glucose_level']
        a=str(request.form['gender'])
        b=str(request.form['ever_married'])
        c=str(request.form['residence'])
        d=str(request.form['work_type'])
        e=str(request.form['smoking_status'])
        f=int(request.form['age'])
        g=int(request.form['heart_disease'])
        h=int(request.form['hypertension'])
        i=float(request.form['bmi'])
        j=float(request.form['glucose_level'])
        le1=LabelEncoder()
        le2=LabelEncoder()
        le3=LabelEncoder()
        le4=LabelEncoder()
        le5=LabelEncoder()
        le1.classes_=np.load('gender.npy',allow_pickle=True)
        le2.classes_=np.load('ever_married.npy',allow_pickle=True)
        le3.classes_=np.load('Residence_type.npy',allow_pickle=True)
        le4.classes_=np.load('work_type.npy',allow_pickle=True)
        le5.classes_=np.load('smoking_status.npy',allow_pickle=True)

        input=np.array([le1.transform([a]),le2.transform([b]),le3.transform([c]),le4.transform([d]),le5.transform([e]),
                        f,g,h,i,j])
        input=input.reshape(1,-1)
        sr=pickle.load(open('sr.pkl','rb'))
        inputs=sr.transform(input)

        model=pickle.load(open('model_s.pkl','rb'))
        results=model.predict(inputs)

        if results==0:
            return redirect(url_for("result",resu="you are fit and fine"))
        else:
            return redirect(url_for("result",resu="you do not have a high chance for stroke"))
    else:
        return render_template('index.html')

@app.route('/<resu>')
def result(resu):
    return render_template('results.html',prediction_text=resu)

if '__main__'==__name__:
    app.run(debug=True)