from django.db import models
from django.contrib.auth.models import User

sentence = ['sentence1','sentenec2','sentence3']

class KeystrokeDB(models.Model):
	user_name = models.CharField(max_length=30, default=None)
	sentence_index=models.IntegerField(default=0)
	for i in range(10):
		exec("chr%d_up=models.IntegerField(default=0)" % i)
		exec("chr%d_dn=models.IntegerField(default=0)" % i)




class user_list(models.Model):
	user_name = models.CharField(max_length=30, default=None)
	index = models.IntegerField(default=0)

