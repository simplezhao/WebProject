from django.urls import path,include
from io126 import views

urlpatterns = [path('in_126/',views.in_126,name='in_126'),
               path('out_126/',views.out_126,name='out_126'),]
