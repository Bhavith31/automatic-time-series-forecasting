from django.urls import path
from . import views

urlpatterns = [
    path('',                                    views.index,          name='index'),
    path('forecast/',                           views.run_forecast,   name='run_forecast'),
    path('result/<int:forecast_id>/',           views.show_result,    name='show_result'),
    path('history/',                            views.history,        name='history'),
    path('dashboard/',                          views.dashboard,      name='dashboard'),
    path('about/',                              views.about,          name='about'),
    path('download/<int:forecast_id>/',         views.download_result,name='download_result'),
    path('delete/<int:forecast_id>/',           views.delete_forecast,name='delete_forecast'),
    path('api/chart-data/<int:forecast_id>/',   views.get_chart_data, name='get_chart_data'),
]
