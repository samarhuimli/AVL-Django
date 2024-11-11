from django.urls import path

from AI.views import AI

urlpatterns = [
    path('query/', AI.as_view(), name='ai_query'),
]
