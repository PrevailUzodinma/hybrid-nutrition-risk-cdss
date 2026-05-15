from django.urls import path
from . import view

app_name = "patients"

urlpatterns = [
    path("patient/<int:patient_id>/", view.patient_record, name="patient_record"),
]