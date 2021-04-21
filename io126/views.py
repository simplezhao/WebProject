from django.shortcuts import render

# Create your views here.
def in_126(request):
     return render(request,"io126/126进.html");
def out_126(request):
     return render(request,"io126/126出.html");