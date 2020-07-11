from django.shortcuts import render
from django.http import HttpResponse


def index(request):
    context = {}
    return render(request, 'table.html', context)


def predictSales(request):
    context = {"a": "Hello World"}
    if request.method == 'POST':
        print(request.POST.dict())
        temp = {}
        temp['item_no'] = request.POST.get("item_no")
        temp['sales'] = request.POST.get("sales")
        temp['date'] = request.POST.get("date")
    return render(request, "table.html", context)


def add(request):
    context = {'a': 'HelloWorld!'}
    return render(request, 'add.html', context)
