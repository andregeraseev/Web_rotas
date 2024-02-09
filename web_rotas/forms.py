from allauth.account.forms import SignupForm
from django import forms
# mysite/forms.py
from allauth.account.forms import SignupForm

class MyCustomSignupForm(SignupForm):

    def save(self, request):
        user = super(MyCustomSignupForm, self).save(request)
        # Adicione sua lógica personalizada aqui
        return user

# mysite/forms.py
from allauth.account.forms import LoginForm

class MyCustomLoginForm(LoginForm):
    def login(self, *args, **kwargs):
        # Adicione sua lógica personalizada aqui
        return super(MyCustomLoginForm, self).login(*args, **kwargs)

    def __init__(self, *args, **kwargs):
        super(MyCustomLoginForm, self).__init__(*args, **kwargs)
        # Remova o campo 'remember' se existir
        self.fields.pop('remember', None)