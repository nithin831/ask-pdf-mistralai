from django import forms
from .models import Pdf
from django.core.exceptions import ValidationError


class PdfForm(forms.ModelForm):
    class Meta:
        model = Pdf
        fields = ['file']
        widgets = {
            'file': forms.FileInput(attrs={'class': 'form-control-file', 'accept': '.pdf'})
        }
        
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Hide label
        self.fields['file'].label = ''
    def clean_file(self):
        file = self.cleaned_data.get('file', False)
        if file:
            if not file.name.endswith('.pdf'):
                raise ValidationError("Only PDF files are allowed.")
        return file