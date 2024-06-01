from django.contrib import admin
from django.urls import path
from Performance.views import *

urlpatterns = [
    path('admin/', admin.site.urls),
    path('index/', index, name='index'),
    path('result/', result, name='/result/'),  # Add URL pattern for the result view
]
