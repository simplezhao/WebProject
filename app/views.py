"""
Definition of views.
"""
#!/usr/bin/python
# -*- coding: UTF-8 -*-
from datetime import datetime
from django.shortcuts import render
from django.http import HttpRequest,JsonResponse
from .models import Workdays2020
import json

def home(request):
    """Renders the home page."""
    assert isinstance(request, HttpRequest)
    return render(
        request,
        'app/index.html',
        {
            'title':'Home Page',
            'year':datetime.now().year,
        }
    )

def contact(request):
    """Renders the contact page."""
    assert isinstance(request, HttpRequest)
    return render(
        request,
        'app/contact.html',
        {
            'title':'Contact',
            'message':'Your contact page.',
            'year':datetime.now().year,
        }
    )

def about(request):
    """Renders the about page."""
    assert isinstance(request, HttpRequest)
    return render(
        request,
        'app/about.html',
        {
            'title':'About',
            'message':'Your application description page.',
            'year':datetime.now().year,
        }
    )


def analysis(request):
    return render(request, 'app/analysis.html')

def getCalendarData(request):
    data = models.Workdays2020.objects.values('date', 'value')
    json_data = serializers.serialize('json', data)
    return JsonResponse(list(json_data), safe=False)