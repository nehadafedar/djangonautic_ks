
from django.contrib import admin
from django.urls import path, include
from . import views
from django.contrib.staticfiles.urls import staticfiles_urlpatterns

urlpatterns = [
    path('admin/', admin.site.urls),
    path('mainpage/',include('mainpage.urls'), name="mainpage"),
    path('Signup/',views.signup_view, name="Signup"),
    path('Login/',views.login_view,name="Login"),
    path('homepage/',views.homepage,name="homepage"),
    path('logout/',views.logout_view,name="logout"),  
    path('thankyou/',views.thankyou_view,name="thankyou"),
    path('',views.homepage,name="homepage"),
]

urlpatterns += staticfiles_urlpatterns()