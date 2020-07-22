from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('', include(('main.urls', 'MoveCounter'), namespace='main')),
    path('admin/', admin.site.urls),
]
