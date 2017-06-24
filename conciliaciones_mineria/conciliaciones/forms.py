from .models import modelLey
from django import forms


class formLey(forms.ModelForm):
    class Meta:
        model = modelLey
        fields = "__all__"
