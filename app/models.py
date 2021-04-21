# This is an auto-generated Django model module.
# You'll have to do the following manually to clean this up:
#   * Rearrange models' order
#   * Make sure each model has one field with primary_key=True
#   * Make sure each ForeignKey has `on_delete` set to the desired behavior.
#   * Remove `managed = False` lines if you wish to allow Django to create, modify, and delete the table
# Feel free to rename the models, but don't rename db_table values or field names.
from __future__ import unicode_literals

from django.db import models


class Station(models.Model):
    station_id = models.CharField(primary_key=True, max_length=255)
    station_name = models.CharField(max_length=255)
    route = models.CharField(max_length=255)
    vegion = models.CharField(max_length=255, blank=True, null=True,verbose_name = '行政区域')

    class Meta:
        db_table = 'station'
        verbose_name_plural = '站点'


class Trips(models.Model):
    trip_id = models.AutoField(primary_key=True)
    user = models.ForeignKey('Users', models.DO_NOTHING)
    in_name = models.CharField(max_length=255,verbose_name = '入站名')
    date_in = models.CharField(max_length=255,verbose_name = '入站时间')
    out_name = models.CharField(max_length=255,verbose_name = '出站名')
    date_out = models.CharField(max_length=255,verbose_name = '出站时间')
    channel_id = models.IntegerField(verbose_name = '渠道类型')
    price = models.IntegerField(verbose_name = '票价')

    class Meta:
        db_table = 'trips'
        verbose_name_plural = '乘车记录'


class Users(models.Model):
    user_id = models.CharField(primary_key=True, max_length=255)
    area = models.CharField(max_length=255)
    birth_year = models.CharField(max_length=255)
    sex = models.CharField(max_length=255)

    class Meta:
        db_table = 'users'
        verbose_name_plural = '用户'


class Workdays2020(models.Model):
    date = models.CharField(primary_key=True, max_length=255)  # Field renamed because it wasn't a valid Python identifier.
    value = models.CharField(max_length=255)  # Field renamed because it wasn't a valid Python identifier.

    class Meta:
        db_table = 'workdays2020'

