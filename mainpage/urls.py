from django.contrib import admin
from django.urls import path
from . import views

app_name='mainpage'
urlpatterns = [
	path('',views.mainpage_view,name="mainpage"),
	path('update/',views.update,name="update"),

	]