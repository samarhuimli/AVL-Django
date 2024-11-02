from django.shortcuts import render

# Create your views here.
from django.shortcuts import render
from .forms import DocumentSearchForm
from .models import Document

def search_and_upload(request):
    if request.method == 'POST':
        form = DocumentSearchForm(request.POST, request.FILES)
        if form.is_valid():
            # Logique pour gérer le fichier uploadé
            file = form.cleaned_data['file']
            Document.objects.create(file=file)
            return render(request, 'upload_success.html')  # Page de succès ou de confirmation
    else:
        form = DocumentSearchForm()
    return render(request, 'search_upload.html', {'form': form})
