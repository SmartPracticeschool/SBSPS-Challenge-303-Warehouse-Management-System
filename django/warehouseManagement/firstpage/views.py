from django.shortcuts import render
from django.http import HttpResponse
from sales import callingFunction
from .models import Sales
#os.system('pip install scikit-learn')
from datetime import datetime
from datetime import timedelta as td
import datetime as dt
import pandas as pd
from  json import dumps

def index(request):
    #context = {}
    db_data_list = []
    test = Sales.objects.all()
    for record in test:
        json_obj = dict(item_no=record.item_no, date=record.date, sales=record.sales,)
        db_data_list.append(json_obj)

    context = {"data": db_data_list}
    #item1()
    return render(request, 'table.html', context)


def predictSales(request):
    if request.method == 'POST':
        temp = {}
        temp['item_no'] = request.POST.get("item_no")
        temp['sales'] = request.POST.get("sales")
        temp['date'] = request.POST.get("date")
    
    date_1 = datetime.strptime(temp['date'], '%Y-%m-%d').date()
    end_date = date_1 + td(days=7)    
    result = callingFunction(int(temp['item_no']), temp['date'], int(temp['sales']),end_date)
    
    db_data_list = []
    test = Sales.objects.all()
    for record in test:
        if record.item_no == temp['item_no'] :
            Sales.objects.filter(item_no = record.item_no).update(sales=result['sales'],date=result['date'])
            json_obj = dict(item_no=temp['item_no'], date=str(result['date']), sales=result['sales'])
            db_data_list.append(json_obj)
        else:  
            json_obj = dict(item_no=record.item_no, date=record.date, sales=record.sales,)
            db_data_list.append(json_obj)

    #print(db_data_list)
    context = {"data": db_data_list}

    return render(request, "table.html", context)


def add(request):
    context = {'a': 'HelloWorld!'}
    return render(request, 'add.html', context)


def item1(request):
    if request.method == 'POST':
        item_no = request.POST.get("item_no")

    df = pd.read_csv("D:\IBM_Hack_2020/Warehouse_train_copy.csv", usecols=['item','sales','date'])
    item_df = df[df.item == int(item_no)]
    item_df.date = item_df.date.apply(lambda x: str(str(x)[0:4])+"-"+str(dt.date(int(str(x)[0:4]), int(str(x)[5:7]), int(str(x)[8:10])).isocalendar()[1]).zfill(2))
    item_group = item_df.groupby(['date'])['sales'].sum().reset_index()
    sales_list = item_group['sales'].tolist()
    dates_list = item_group['date'].tolist()
    salesJSON = dumps(sales_list)
    datesJSON = dumps(dates_list)
    pred = Sales.objects.filter(item_no = item_no)
    for record in pred:
        json_obj = dict(item_no=record.item_no, date=record.date, sales=record.sales,)
        x = json_obj['date']
        date1 = str(str(x)[0:4])+"-"+str(dt.date(int(str(x)[0:4]), int(str(x)[5:7]), int(str(x)[8:10])).isocalendar()[1]).zfill(2)

    context = {"sales_list":salesJSON,"dates_list":datesJSON,"item_no":item_no, "date_pred": date1,"sales_pred":json_obj["sales"]}
    return render(request, 'item1.html', context)

def about(request):
    context = {'a': 'HelloWorld!'}
    return render(request, 'about.html', context)
