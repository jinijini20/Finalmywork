from django.urls import path
from . import views

urlpatterns = [
      # 메인 페이지
    path('', views.register, name='register'),  # /about 페이지
    path('myaccountpage', views.myaccount_page, name='myaccountpage'),
]
