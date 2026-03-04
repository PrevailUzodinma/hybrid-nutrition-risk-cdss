from django.urls import path
from . import views

app_name = "patients"

urlpatterns = [
    path("patient/<int:patient_id>/", views.patient_record, name="patient_record"),
]