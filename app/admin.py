from django.contrib import admin

# Register your models here.
from .models import Trips,Station,Workdays2020,Users

admin.site.register(Trips)
admin.site.register(Station)
admin.site.register(Workdays2020)
admin.site.register(Users)
