from django.db import models

# Create your models here.
class Sheet1(models.Model):
    time = models.CharField(primary_key=True, max_length=255)
    number_126_in = models.IntegerField(db_column='126_in', blank=True, null=True)  # Field renamed because it wasn't a valid Python identifier.
    number_126_out = models.IntegerField(db_column='126_out', blank=True, null=True)  # Field renamed because it wasn't a valid Python identifier.

    class Meta:
        db_table = 'sheet1'
        verbose_name_plural = '126进出情况'

class Sheet2(models.Model):
    day_id = models.CharField(primary_key=True, max_length=255)
    weather = models.CharField(max_length=255, blank=True, null=True)
    tmax = models.CharField(db_column='Tmax', max_length=255, blank=True, null=True)  # Field name made lowercase.
    tmin = models.CharField(db_column='Tmin', max_length=255, blank=True, null=True)  # Field name made lowercase.

    class Meta:
        db_table = 'sheet2'
        verbose_name_plural = '城市天气数据'