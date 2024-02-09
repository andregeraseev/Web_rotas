# forms.py
from django import forms

class EnderecoForm(forms.Form):
    endereco = forms.CharField(label="Endereço", widget=forms.TextInput(attrs={'placeholder': 'Endereço'}))


class ParametrosForm(forms.Form):
   max_stops_input = forms.IntegerField(label="Número máximo de paradas", widget=forms.NumberInput(attrs={'placeholder': 'Número máximo de paradas'}))

   max_time_input = forms.IntegerField(label="Tempo máximo de viagem", widget=forms.NumberInput(attrs={'placeholder': 'Tempo máximo de viagem'}))


class TecnicoForm(forms.Form):

    partida = forms.CharField(label="Endereço de Partida", widget=forms.TextInput(attrs={'placeholder': 'Partida'}))
    final = forms.CharField(label="Endereço de Final", widget=forms.TextInput(attrs={'placeholder': 'Final'}))


from django.forms import formset_factory

EnderecoFormSet = formset_factory(EnderecoForm, extra=1)
TecnicoFormSet = formset_factory(TecnicoForm, extra=1)

