from django.urls import path,include
from io126 import views

urlpatterns = [path('in_126/',views.in_126,name='in_126'),]
