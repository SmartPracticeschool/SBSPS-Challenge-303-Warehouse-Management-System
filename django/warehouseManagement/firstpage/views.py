from django.shortcuts import render
from django.http import HttpResponse
from sales import callingFunction
#os.system('pip install scikit-learn')


def index(request):
    context = {}
    return render(request, 'table.html', context)


def predictSales(request):
    if request.method == 'POST':
        print(request.POST.dict())
        temp = {}
        temp['item_no'] = request.POST.get("item_no")
        temp['sales'] = request.POST.get("sales")
        temp['date'] = request.POST.get("date")
    result = {}

    result = callingFunction(
        int(temp['item_no']), temp['date'], int(temp['sales']), "2018-02")
    print(result)
    return render(request, "table.html", result)


def add(request):
    context = {'a': 'HelloWorld!'}
    return render(request, 'add.html', context)
