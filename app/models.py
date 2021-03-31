# This is an auto-generated Django model module.
# You'll have to do the following manually to clean this up:
#   * Rearrange models' order
#   * Make sure each model has one field with primary_key=True
#   * Make sure each ForeignKey has `on_delete` set to the desired behavior.
#   * Remove `managed = False` lines if you wish to allow Django to create, modify, and delete the table
# Feel free to rename the models, but don't rename db_table values or field names.
from __future__ import unicode_literals

from django.db import models


class Errordata(models.Model):
    user_id = models.CharField(max_length=255, blank=True, null=True)
    in_name = models.CharField(max_length=255, blank=True, null=True)
    in_time = models.CharField(max_length=255, blank=True, null=True)
    out_name = models.CharField(max_length=255, blank=True, null=True)
    out_time = models.CharField(max_length=255, blank=True, null=True)
    route_id = models.CharField(max_length=255, blank=True, null=True)
    price = models.CharField(max_length=255, blank=True, null=True)

    class Meta:
        db_table = 'errordata'


class Station(models.Model):
    id = models.CharField(primary_key=True, max_length=255)
    station_name = models.CharField(max_length=255)
    route = models.CharField(max_length=255)
    vegion = models.CharField(max_length=255, blank=True, null=True)

    class Meta:
        db_table = 'station'


class Trips(models.Model):
    user = models.ForeignKey('Users', models.DO_NOTHING, blank=True, null=True)
    in_name = models.CharField(max_length=255)
    in_time = models.CharField(max_length=255)
    out_name = models.CharField(max_length=255)
    out_time = models.CharField(max_length=255)
    channel_id = models.CharField(max_length=255)
    price = models.CharField(max_length=255)

    class Meta:
        db_table = 'trips'


class Users(models.Model):
    user_id = models.CharField(primary_key=True, max_length=255)
    area = models.CharField(max_length=255)
    birth_year = models.CharField(max_length=255)
    sex = models.CharField(max_length=255)

    class Meta:
        db_table = 'users'


class Workdays2020(models.Model):
    number_20200101 = models.CharField(db_column='20200101', primary_key=True, max_length=255)  # Field renamed because it wasn't a valid Python identifier.
    number_3 = models.CharField(db_column='3', max_length=255)  # Field renamed because it wasn't a valid Python identifier.

    class Meta:
        db_table = 'workdays2020'
