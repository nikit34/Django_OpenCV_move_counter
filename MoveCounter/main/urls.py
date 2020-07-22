from django.urls import path

from .views import Main


urlpatterns = [
    path('main/', Main.as_view(), name='main')
]