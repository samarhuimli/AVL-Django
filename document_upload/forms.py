# forms.py
from django import forms

class DocumentSearchForm(forms.Form):
    search = forms.CharField(
        max_length=100,
        required=False,
        widget=forms.TextInput(attrs={
            'placeholder': 'Rechercher ou télécharger un fichier...',
            'class': 'search-bar'
        })
    )
    file = forms.FileField(required=True)
