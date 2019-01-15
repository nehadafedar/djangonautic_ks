from django.http import HttpResponse
from django.shortcuts import render,redirect
from django.contrib.auth.forms import UserCreationForm,AuthenticationForm
from django.contrib.auth import login,logout
from mainpage.models import user_list



sentence = ['sentence1','sentence2','sentence3']
index=0

def homepage(request):
	return render(request,'homepage.html')

def signup_view(request):
	global index

	if request.method == 'POST':
		
		form = UserCreationForm(request.POST)
		if form.is_valid():
			user = form.save()
			#log the user in
			login(request, user)
			print("in Signup post")
			name=user_list.objects.create(user_name=str(user), index=0)
			name.save()

			
			return redirect('mainpage:mainpage')
	else:
		form = UserCreationForm()
	return render(request, 'Signup.html',{'form':form})

def login_view(request):
	global index

	if request.method == 'POST':
		form = AuthenticationForm(data=request.POST)
		if form.is_valid():
			#log in the user
			user = form.get_user()
			login(request, user)
			
			name=user_list.objects.get(user_name=str(user))
			index=name.index

			return redirect('mainpage:mainpage')

			
	else:
		form = AuthenticationForm()
	return render(request,'Login.html',{'form':form})


def logout_view(request):
	if request.method=='POST':
		logout(request)
		return render(request, 'homepage.html')

def thankyou_view(request):
	return render(request,'thankyou.html')


