from __future__ import unicode_literals
from django.db import models
# Create your models here.

class Sales(models.Model):
	item_no = models.CharField(max_length = 3)
	date  = models.CharField(max_length = 10)
	sales = models.CharField(max_length = 10)

#If you change this, do makemigrations and then do migrate