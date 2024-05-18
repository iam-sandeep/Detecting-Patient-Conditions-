from django.contrib import admin
from django.urls import path  , include
from . import views
urlpatterns = [
   
    
    path('',views.SignupPage,name='signup'),
    path('login/',views.LoginPage,name='login'),
    path('logout/',views.LogoutPage,name='logout'),
    path('profile/',views.ProfilePage,name='profile'),

    path('home/',views.Home,name='home'),

    path('heart/', views.heart, name="heart"),
    # path('heart/heartsolution/', views.solution, name='solution'),
    # path('heart/congratulations/', views.congratulations, name='congratulations'),

    path("diabaties/", views.diabaties,name="diabaties"),
    path("diabaties/result/", views.result,name="result"),

    path('cancer', views.cancer, name="cancer"),
    # path('cancer/csolution/', views.cansolution, name='cansolution'),
    # path('cancer/congratulations/', views.cancongratulations, name='cancongratulations'),



    
]

