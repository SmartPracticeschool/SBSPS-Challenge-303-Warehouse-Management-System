"""warehouseManagement URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from django.conf.urls import url
from firstpage import views

urlpatterns = [
    path('admin/', admin.site.urls),
    url('^$', views.index, name="table"),
    url('predictSales', views.predictSales, name='PredictSales'),
    url('add', views.add, name='add'),
    url('item1', views.item1, name='item1'),
    url('about', views.about, name="about"),

]
