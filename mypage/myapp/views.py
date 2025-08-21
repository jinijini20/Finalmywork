from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse

def register(request):
    return render(request,'myapp/register.html')


def myaccount_page(request):
    return render(request, 'myapp/myaccountpage.html')