from django.urls import path,include
from ob_predict import views

urlpatterns = [path('health/',views.check,name='check'),
               path('kalman-od-predict/',views.inference,name='inference'),]

