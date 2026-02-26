# appointments/tests/test_views.py
from django.test import TestCase, Client
from django.urls import reverse
from django.contrib.auth.models import User

class AppointmentListViewTest(TestCase):
    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user("testdoc", password="pass")

    def test_redirect_if_unauthenticated(self):
        response = self.client.get(reverse("appointments:appointment_list"))
        self.assertEqual(response.status_code, 302)
        self.assertIn("/accounts/login/", response["Location"])

    def test_200_when_logged_in(self):
        self.client.login(username="testdoc", password="pass")
        response = self.client.get(reverse("appointments:appointment_list"))
        self.assertEqual(response.status_code, 200)

    def test_correct_template(self):
        self.client.login(username="testdoc", password="pass")
        response = self.client.get(reverse("appointments:appointment_list"))
        self.assertTemplateUsed(response, "appointments/appointment_list.html")