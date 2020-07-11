from django.shortcuts import render
from django.http import HttpResponse


def index(request):
	context = {'a':'HelloWorld!'}
	return render(request, 'index.html', context)

def predictSales(request):
	context = {'a':'HelloWorld!'}
	return render(request, 'add.html', context)

def add(request):
	context = {'a':'HelloWorld!'}
	return render(request, 'add.html', context)
