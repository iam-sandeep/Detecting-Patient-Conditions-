from django.shortcuts import render,HttpResponse,redirect
from django.contrib.auth.models import User
from django.contrib.auth import authenticate,login,logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages
import json
from django.core.serializers.json import DjangoJSONEncoder
# Create your views here.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score



from django.shortcuts import render
from django.http import HttpResponse
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from Disease.forms import  HeartDiseaseForm
# Create your views here.
#pdf
from django.http import FileResponse
from xhtml2pdf import pisa
from io import BytesIO
from django.template.loader import render_to_string


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from django.shortcuts import render
from .forms import HeartDiseaseForm

from Disease.cancer import CancerDiseaseForm


# from Disease.diabeteseform import DiabetesForm




# @login_required(login_url='login')
@login_required(login_url='login')
def Home(request):
    return render (request,'home.html')


def SignupPage(request):
    
    if request.method=='POST':
        
        uname=request.POST.get('username')
        email=request.POST.get('email')
        pass1=request.POST.get('password1')
        pass2=request.POST.get('password2')

        user_exist=User.objects.filter(username = uname).exists()
        email_check=User.objects.filter(email=email).exists()
   
        
        if pass1!=pass2:
            return HttpResponse("Your password and confrom password are not Same!!")
        
        elif user_exist==True:
            messages.error(request, "This username is already taken")
            return redirect('/')
        
        elif email_check==True:
            messages.error(request, "This email is already taken")
            return redirect('/')
        
        
        
        else:
            my_user=User.objects.create_user(username = uname,email=email,password=pass1)
        
            my_user.save()
            return redirect('login')
    return render (request,'signup.html')
        
        
        

def LoginPage(request):
    
    if request.method=='POST':
        username=request.POST.get('username')
        pass1=request.POST.get('pass')
        # request.session['username']=username
        user=authenticate(request,username=username,password=pass1)
        if user is not None:
            login(request,user)
            return redirect('home')
        else:
            return HttpResponse ("Username or Password is incorrect!!!")
    return render(request,'login.html')

def LogoutPage(request):
    logout(request)
    return redirect('login')


def ProfilePage(request):
    user = User.objects.filter(username=request.user).values("diabities","cancer","heart")
    # print(user[0]["diabities"], "This is user")
    return render(request,'profile.html',{"cancer":user[0]["cancer"],"diabities":user[0]["diabities"],"heart":user[0]["heart"]})

# Diabetes Start
def diabaties(request):
    return render(request, 'diabaties.html')
def result(request):
    data = pd.read_csv("static/diabetes.csv")
    X= data.drop("Outcome",axis=1) 
    Y= data["Outcome"]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    model = LogisticRegression()
    model.fit(X_train, Y_train)

    val1 = float(request.GET['n1'])
    val2 = float(request.GET['n2'])
    val3 = float(request.GET['n3'])
    val4 = float(request.GET['n4'])
    val5 = float(request.GET['n5'])
    val6 = float(request.GET['n6'])
    val7 = float(request.GET['n7'])
    val8 = float(request.GET['n8'])

    pred = model.predict([[val1, val2, val3, val4, val5, val6, val7, val8]])

    result1 = ""
    value = 0

    if pred == [1]:
        value = 1
        result1 = "Algorithm in process."
    else:
        value = 0
        result1 = "Algorithm in process."

    if value == 1:
        User.objects.filter(username=request.user).update(diabaties=True)
        return render(request, "result.html", {"result2": result1})
    else:
        User.objects.filter(username=request.user).update(diabities=False)
        return render(request, "algorithm.html", {"result2": result1})


#Diabetse End Here


#Heart start Here
def heart(request):
    df = pd.read_csv('static/heart.csv')
    # Replace the rest of the old code with the new code

    # Check for missing values
    df = df.fillna(df.median())

    # Drop duplicates
    df = df.drop_duplicates()

    # Normalize the data
    scaler = StandardScaler()
    df[df.columns[:-1]] = scaler.fit_transform(df[df.columns[:-1]])

    # Split into X and y
    X = df.drop('target', axis=1)
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    value = ''
    if request.method == 'POST':

        # Get user data from form
        user_data = np.array([
            float(request.POST[field])
            for field in ('age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                          'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal')
        ]).reshape(1, 13)

        # Scale user data
        user_data = scaler.transform(user_data)

        # Make predictions
        predictions = model.predict(user_data)
        # temp=int(predictions[0])
        # if temp == 1:
        #     # Redirect to the solution page if predicted to have heart disease
        #     return redirect('solution')
        # elif temp == 0:
        #     # Redirect to the congratulations page if predicted to not have heart disease
        #     return redirect('congratulations')
        result1= ""
        val=1
        if int(predictions[0])==0:
            val=1
            result1="Oops! You have Heart Disease."
        elif int(predictions[0])==1:
            val=0
            result1="Great! You DON'T have Heart Disease."
        if val==1:
            User.objects.filter(username=request.user).update(heart=True)
            return render(request, "heartsolution.html", {"result2": result1})
        else:
            User.objects.filter(username=request.user).update(heart=False)
            return render(request,"congratulations.html",{"result2": result1})
        
    return render(request,
                  'heart.html',
                  {
                      'context': value,
                      'title': 'Heart Disease Prediction',
                      'active': 'btn btn-success peach-gradient text-white',
                      'heart': True,
                      'form': HeartDiseaseForm(),
                  })
def solution(request):
    return render(request, 'heartsolution.html')

def congratulations(request):
    return render(request, 'congratulations.html')

# Heart End Here

#Cancer start here

def cancer(request):
   
    df = pd.read_csv('static/cancer.csv')
   
    features = df.drop(['id','diagnosis'], axis = 1)
    labels = df['diagnosis']
    X=features
    Y=labels

    print(X)
    print(Y) 
 
    value = ''
 
    if request.method == 'POST':
 
        id = float(request.POST['id'])
        diagnosis = float(request.POST['diagnosis'])
        radius_mean = float(request.POST['radius_mean'])
        texture_mean = float(request.POST['texture_mean'])
        perimeter_mean= float(request.POST['perimeter_mean'])
        area_mean= float(request.POST['area_mean'])
        smoothness_mean = float(request.POST['smoothness_mean'])
        concavity_mean= float(request.POST['concavity_mean'])
        concave_points_mean= float(request.POST['concave_points_mean'])
        symmetry_mean= float(request.POST['symmetry_mean'])
        fractal_dimension_mean= float(request.POST['fractal_dimension_mean'])
        radius_se= float(request.POST['radius_se'])
        texture_se = float(request.POST['texture_se'])
        perimeter_se=float(request.POST['perimeter_se'])
        area_se=float(request.POST['area_se'])
        smoothness_se=float(request.POST['smoothness_se'])
        compactness_se=float(request.POST['compactness_se'])
        concavity_se=float(request.POST['concavity_se'])
        concave_points_se=float(request.POST['concave_points_se'])
        symmetry_se=float(request.POST['symmetry_se'])
        fractal_dimension_se=float(request.POST['fractal_dimension_se'])
        radius_worst=float(request.POST['radius_worst'])
        texture_worst=float(request.POST['texture_worst'])
        perimeter_worst=float(request.POST['perimeter_worst'])
        area_worst=float(request.POST['area_worst'])
        smoothness_worst=float(request.POST['smoothness_worst'])
        compactness_worst=float(request.POST['compactness_worst'])
        concavity_worst=float(request.POST['concavity_worst'])
        concave_points_worst=float(request.POST['concave_points_worst'])
        symmetry_worst=float(request.POST['symmetry_worst'])
        fractal_dimension_worst=float(request.POST['fractal_dimension_worst'])

 
        # user_data = np.array(
        #     (id,diagnosis,radius_mean,texture_mean,perimeter_mean,area_mean,smoothness_mean,concavity_mean,concave_points_mean,symmetry_mean,fractal_dimension_mean,radius_se,texture_se,
        #      perimeter_se,area_se,smoothness_se,compactness_se,concavity_se,concave_points_se,symmetry_se,fractal_dimension_se,radius_worst,texture_worst,perimeter_worst,area_worst
        #      ,smoothness_worst,compactness_worst,concavity_worst,concave_points_worst,symmetry_worst,fractal_dimension_worst)
        # ).reshape(1, 31)

        user_data = np.array(
            (radius_mean,texture_mean,perimeter_mean,area_mean,smoothness_mean,concavity_mean,concave_points_mean,symmetry_mean,fractal_dimension_mean,radius_se,texture_se,
            perimeter_se,area_se,smoothness_se,compactness_se,concavity_se,concave_points_se,symmetry_se,fractal_dimension_se,radius_worst,texture_worst,perimeter_worst,area_worst
            ,smoothness_worst,compactness_worst,concavity_worst,concave_points_worst,symmetry_worst,fractal_dimension_worst, 0)
        ).reshape(1, 30)

        print(user_data)

        rf = RandomForestClassifier(
            n_estimators=16,
            criterion='entropy',
            max_depth=9
        )

        rf.fit(np.nan_to_num(X), Y)
        rf.score(np.nan_to_num(X), Y)
        predictions = rf.predict(user_data)
        # print(int(float(predictions[0])))
        # if int(float(predictions[0])) == 1:
        #     # Redirect to the solution page if predicted to have heart disease
        #     return redirect('cansolution')
        # elif int(float(predictions[0])) == 0:
        #     # Redirect to the congratulations page if predicted to not have heart disease
        #     return redirect('cancongratulations')
        result1= ""
        val=1
        if int(float(predictions[0]))==1:
            val=1
            result1="Oops! You have Cancer."
        elif int(float(predictions[0]))==0:
            val=0
            result1="Great! You DON'T have Cancer."
        if val==1:
            User.objects.filter(username=request.user).update(cancer=True)
            return render(request, "csolution.html", {"result2": result1})
        else:
            User.objects.filter(username=request.user).update(cancer=False)
            return render(request,"congratulations.html",{"result2": result1})
    

    return render(request,
                  'cancer.html',
                  {
                      'context': value,
                      'title': 'Cancer Disease Prediction',
                      'active': 'btn btn-success peach-gradient text-white',
                      'heart': True,
                      'form': CancerDiseaseForm(),
                  })
# def cansolution(request):
#     return render(request, 'csolution.html')

# def cancongratulations(request):
#     return render(request, 'congratulations.html')

#Cancer end here